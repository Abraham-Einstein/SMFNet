import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numpy as np
from utils.graph import GraphReasoning
from torchsummary import summary
from thop import profile

# ------------------------------------------------------------------- #

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


########### local enhanced feed-forward network #############
class LeFF(nn.Module):
    def __init__(self, dim=64, hidden_dim=64, act_layer=nn.GELU):
        super(LeFF, self).__init__()

        self.linear1 = nn.Sequential(nn.Conv2d(dim, hidden_dim, kernel_size=1, stride=1, padding=0),
                                     act_layer()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2 = nn.Sequential(nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=0),
                                     act_layer()
        )
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        origin_x = x
        x = self.linear1(x)
        x = self.dwconv(x)
        x = self.linear2(x) + origin_x
        return x


class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(BaseFeatureExtraction, self).__init__()
        # self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1)
        self.norm1 = LayerNorm(dim, 'WithBias')
        # self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.attn = Attention(dim=dim, num_heads=num_heads,
                               bias=False)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = LeFF(dim, hidden_dim=64)
        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))               # leff------------wly
        return x

#####################################################################


class Cross_Trans_Attention(nn.Module):
    def __init__(self,num_heads,dim):
        super(Cross_Trans_Attention, self).__init__()
        self.num_heads = num_heads
        bias=True
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim*1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, feat_guide,feat_op):
        # feat_ref: Guidance
        # feat_ext: Value
        b,c,h,w = feat_guide.shape

        q = self.q_dwconv(self.q(feat_guide))
        kv = self.kv_dwconv(self.kv(feat_op))
        k,v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp=32, oup=32, expand_ratio=2):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)
#
class INV_block_affine(nn.Module):
    def __init__(self, subnet_constructor=InvertedResidualBlock, clamp=2.0, harr=True, in_1=8, in_2=8, imp_map=False):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1)

        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                            stride=1, padding=0, bias=True)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def separateFeature(self, x):

        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2

    def forward(self, x1, x2, rev=True):

        x1, x2 = self.separateFeature(
                    self.shffleconv(torch.cat((x1, x2), dim=1)))

        if not rev:

            t2 = self.f(x2)
            s2 = self.p(x2)
            y1 = self.e(s2) * x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:  # names of x and y are swapped!

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            s2 = self.p(y2)
            y1 = (x1 - t2) / self.e(s2)

        return y1, y2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, dim=64, num_layers=2):
        super(DetailFeatureExtraction, self).__init__()

        HINNmodules = [INV_block_affine() for _ in range(num_layers)]
        self.net = nn.Sequential(*HINNmodules)
        self.GRDB = GRDB_module()

    def forward(self, x):
        # x = self.reduce_channel(x)
        x = self.GRDB(x)
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)
#
class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction_Encoder(nn.Module):
    def __init__(self, dim=32, num_layers=1):
        super(DetailFeatureExtraction_Encoder, self).__init__()

        self.TCA = Cross_Trans_Attention(num_heads=8, dim=dim)
        self.layer = DetailNode()

    def forward(self, x):

        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        feat1 = self.TCA(feat_guide=z2, feat_op=z1)
        feat2 = self.TCA(feat_guide=z1, feat_op=z2)

        feat1, feat2 = self.layer(feat1, feat2)

        return torch.cat((feat1, feat2), dim=1)


# =============================================================================
import numbers


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


###################################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_leff = LeFF(dim, hidden_dim=64)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn_leff(self.norm2(x))        # -----Leff-----wly

        return x

class TransformerBase(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, qkv_bias=False):
        super(TransformerBase, self).__init__()

        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = FeedForward(dim=dim, ffn_factor=ffn_expansion_factor, bias=False)
        # self.mlp = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=False)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    ##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
##################################################################################


class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)

    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        return x

class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)

########################################################################3
'''
  GRDB--------------------------wly
'''

class GRDB_module(nn.Module):
    def __init__(self,
                 inp_channels=64,
                 out_channels=64,
                 dim=64,
                 depth=4
                 ):
        super(GRDB_module, self).__init__()

        self.conv = ConvLeakyRelu2d(in_channels=64, out_channels=64)
        self.GRDB = RGBD(in_channels=64, out_channels=64)

    def forward(self, feature_ir):
        # conv
        x = self.conv(feature_ir)
        local_feature = self.GRDB(x)

        return local_feature

###########################################################################################

class CGR(nn.Module):
    def __init__(self, n_class=2, n_iter=2, chnn=(64, 64, 64), chnn_side=64, chnn_targ=64, rd_sc=1, dila=(4, 8, 16)):
        super().__init__()
        self.n_graph = 1         # 1
        n_node = len(dila)                     # 3
        # n_node = 1
        self.graph = GraphReasoning(chnn_side, rd_sc, dila, n_iter) # ii-----chnn_side
        C_cat = [nn.Sequential(
            nn.Conv2d(ii//rd_sc*n_node, ii//rd_sc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ii//rd_sc),
            nn.ReLU(inplace=True))
            for ii in (chnn + chnn)]

        self.C_cat = nn.ModuleList(C_cat)
        self.C_cls = nn.Conv2d(chnn_targ * 2, chnn_targ, 1, 1)  # n_class -> 1

    def forward(self, input_ir, input_vi):
        img = input_ir                # ir, vis feature tensor
        depth = input_vi
        nd_rgb, nd_dep, nd_key = None, None, False
        for ii in range(self.n_graph):       # just 1 time
            feat_rgb, feat_dep = self.graph([img, depth, nd_rgb, nd_dep], nd_key)

            feat_rgb = torch.cat(feat_rgb, 1)   # list
            feat_rgb = self.C_cat[ii](feat_rgb)          # leader node----wly

            feat_dep = torch.cat(feat_dep, 1)
            feat_dep = self.C_cat[self.n_graph+ii](feat_dep)


        return feat_rgb, feat_dep

class CGR_backbone(nn.Module):
    def __init__(self, chnn_targ=64):
        super(CGR_backbone, self).__init__()
        self.graph = CGR()
        self.C_cls = nn.Conv2d(chnn_targ * 2, chnn_targ, 1, 1)

    def forward(self, feature_V, feature_I, flag):

        if flag:
            feat_I, feat_V = self.graph(feature_I, feature_V)
            feat = torch.cat((feat_I, feat_V), dim=1)        # ----------cat---------wly
            out = self.C_cls(feat)
            return out, feat_I, feat_V
        else:
            feat_I, feat_V = self.graph(feature_I, feature_V)
            return feat_I, feat_V

############################################################################################

class Block(nn.Module):
    def __init__(self, dim, num_heads, depth, resi_connection='3conv'):
        super(Block, self).__init__()

        self.dim = dim
        self.residual_group = nn.ModuleList([nn.Sequential(
            TransformerBlock(dim=dim, num_heads=num_heads,
                            bias=False, LayerNorm_type='WithBias'),
            TransformerBase(dim=dim, num_heads=num_heads,
                            ffn_expansion_factor=2, qkv_bias=False)
            )
            for i in range(depth)])

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))


    def forward(self, x):
        orgin = x
        for blk in self.residual_group:
            x = blk(x)
        return self.conv(x) + orgin  # residual connection----------wly

# ----------------------------------------------------------------------- #
'''
2024.1.21 global refinement model----------------------wly
'''

class GRM(nn.Module):  # Global Refinement Module
    def __init__(self):
        super(GRM, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(128, 128, 3, 1, 1)
        self.SFT_scale_conv1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.SFT_shift_conv0 = nn.Conv2d(128, 128, 1)
        self.SFT_shift_conv1 = nn.Conv2d(128, 64, 1)
        self.SFT2 = nn.Conv2d(192, 64, 3, 1, 1)

    def forward(self, detail_feature, base_feature, GCN_feature):
        # x[0]: fea; x[1]: cond
        x = GCN_feature   # 64
        c = torch.cat([detail_feature, base_feature], dim=1)  # 128
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(c), 0.1, inplace=True))
        m = F.leaky_relu(self.SFT2(torch.cat([x, scale], 1)))   # 192------>64------wly
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(c), 0.1, inplace=True)) # 128----->64----wly
        return m + shift

#######################################################################################################



class SwinIR_Encoder(nn.Module):
    def __init__(self, in_chans=64,
                 embed_dim=64, depths=[4, 2], num_heads=[8, 8, 8, 8],
                 window_size=8, mlp_ratio=4., drop_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 upscale=2, img_range=1., upsampler='', resi_connection='3conv'):
        super(SwinIR_Encoder, self).__init__()
        num_in_ch = in_chans
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)  # SFE  96-------64---------wly

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)  # 3
        # self.num_layers = 1
        self.embed_dim = embed_dim  # 96------------64-----------wly
        self.ape = ape
        self.patch_norm = patch_norm  # True
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio  # 4
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.reduce_channel = nn.Conv2d(int(embed_dim * 2), int(embed_dim), kernel_size=1, bias=False)

        # build Restormer Transformer blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Block(dim=embed_dim, num_heads=num_heads[0], depth=depths[1])
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def forward(self, x):
        inp_enc_level1 = self.conv_first(x)
        base_feature = self.forward_features(inp_enc_level1)

        return base_feature

class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[1, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=4,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[3],
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.baseFeature = SwinIR_Encoder()
        self.detailFeature = DetailFeatureExtraction_Encoder()
        self.GRDB_module = GRDB_module()

    def forward(self, inp_img):
        # ---------- restormer block ------------- #
        inp_enc_level1 = self.patch_embed(inp_img)  # shallow feature extractor---wly
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # ---------- detail feature extraction ---------- #
        out_enc_level2 = self.GRDB_module(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level2)


        # ---------- base feature extraction ----------- #
        global_feature = self.baseFeature(out_enc_level1)

        return global_feature, detail_feature, inp_enc_level1, out_enc_level1


class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=64,
                 out_channels=1,
                 dim=64,
                 num_blocks=[1, 2],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=4,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()

        # self.reduce_channel = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[3],
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, int(dim) // 4, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 4, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, feature):

        out_enc_level0 = self.encoder_level1(feature)
        inp_img = None      # ----if inp_img=None------wly
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level0) + inp_img  # ---poor performance in fact------wly
        else:
            out_enc_level1 = self.output(out_enc_level0)   # the better choice-----wly
        return self.sigmoid(out_enc_level1), out_enc_level0



if __name__ == '__main__':
    height = 128
    width = 128
    window_size = 8
    modelE = Restormer_Encoder().cuda()
    modelD = Restormer_Decoder().cuda()

