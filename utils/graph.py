import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.ConvGRU as ConvGRU


class PPM(nn.Module):
    def __init__(self, chnn_in, rd_sc, dila):
        super(PPM, self).__init__()
        chnn = chnn_in // rd_sc
        convs = [nn.Sequential(
            nn.Conv2d(chnn_in, chnn, 3, padding=ii, dilation=ii, bias=False),
            nn.BatchNorm2d(chnn),
            nn.ReLU(inplace=True))
            for ii in dila]                 #  3-------------wly
        self.convs = nn.ModuleList(convs)

    def forward(self, inputs):
        # print("input", inputs.shape)
        feats = []
        for conv in self.convs:
            feat = conv(inputs)
            # print(feat.shape)           # [1, 64, 128, 128]
            feats.append(feat)
        return feats


class GraphReasoning(nn.Module):
    def __init__(self, chnn_in, rd_sc, dila, n_iter):    # 512 256 128
        super().__init__()
        self.n_iter = n_iter
        self.ppm_rgb = PPM(chnn_in, rd_sc, dila)      # rd_sc--------32------wly
        self.ppm_dep = PPM(chnn_in, rd_sc, dila)
        self.n_node = len(dila)       # 3
        self.graph_rgb = GraphModel(self.n_node, chnn_in//rd_sc)
        self.graph_dep = GraphModel(self.n_node, chnn_in//rd_sc)
        chnn = chnn_in * 2 // rd_sc
        C_ca = [nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(chnn, chnn//4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(chnn//4, chnn_in//rd_sc, 1, bias=False))
            for ii in range(2)]
        self.C_ca = nn.ModuleList(C_ca)
        C_pa = [nn.Conv2d(chnn_in//rd_sc, 1, 1, bias=False) for ii in range(2)]
        self.C_pa = nn.ModuleList(C_pa)

    def _enh(self, Func, src, dst):
        out = torch.sigmoid(Func(src)) * dst + dst
        # print(out.shape)   # [1, 8, 192, 256]
        return out

    def _inn(self, Func, feat):
        feat = [fm.unsqueeze(1) for fm in feat]
        feat = torch.cat(feat, 1)
        for ii in range(self.n_iter):
            feat = Func(feat)            # graph model
        feat = torch.split(feat, 1, 1)
        feat = [fm.squeeze(1) for fm in feat]
        return feat

    def _int(self, Func, src_1, src_2):
        out_2 = src_1 * torch.sigmoid(Func[0](src_1 - src_2)) + src_2         #  C_pa[0]
        out_1 = src_2 * torch.sigmoid(Func[1](src_2 - src_1)) + src_1         #  C_pa[1]
        # print(out_1.shape, out_2.shape)   # [1, 16, 96, 128]
        return out_1, out_2

    def forward(self, inputs, node=False):
        feat_rgb, feat_dep, nd_rgb, nd_dep = inputs
        # print(feat_dep.shape)
        feat_rgb = self.ppm_rgb(feat_rgb)        # list------node generation---------wly
        feat_dep = self.ppm_dep(feat_dep)
        if node:
            feat_rgb = [self._enh(self.C_ca[0], nd_rgb, fm) for fm in feat_rgb]  # leader node and feature[i]---wly
            feat_dep = [self._enh(self.C_ca[1], nd_dep, fm) for fm in feat_dep]
        for ii in range(self.n_node):
            # edge generation between ir and vi nodes---wly
            feat_rgb[ii], feat_dep[ii] = self._int([self.C_pa[0], self.C_pa[1]], feat_rgb[ii], feat_dep[ii])
            feat_rgb[ii], feat_dep[ii] = self._int([self.C_pa[0], self.C_pa[1]], feat_rgb[ii], feat_dep[ii])
        feat_rgb = self._inn(self.graph_rgb, feat_rgb)
        feat_dep = self._inn(self.graph_dep, feat_dep)
        return feat_rgb, feat_dep


class GraphModel(nn.Module):
    def  __init__(self, N, chnn_in=256):
        super().__init__()
        self.n_node = N
        chnn = chnn_in
        self.C_wgt = nn.Conv2d(chnn*(N-1), (N-1), 1, groups=(N-1), bias=False)
        self.ConvGRU = ConvGRU.ConvGRUCell(chnn, chnn, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))       # trainable-----wly

    def forward(self, inputs):
        b, n, c, h, w = inputs.shape
        feat_s = [inputs[:,ii,:] for ii in range(self.n_node)]   # edge--------wly
        pred_s =[]
        for idx_node in range(self.n_node):
            h_t = feat_s[idx_node]
            h_t_m = h_t.repeat(1, self.n_node-1, 1, 1)
            h_n = torch.cat([feat_s[ii] for ii in range(self.n_node) if ii != idx_node], dim=1)
            msg = self._get_msg(h_t_m, h_n)   # message delivery among the edges------wly
            m_t = torch.sum(msg.view(b, -1, c, h, w), dim=1)
            h_t = self.ConvGRU(m_t, h_t)
            base = feat_s[idx_node]
            pred_s.append(h_t*self.gamma+base)   # multiply former nodes with leader weight in channel domain--wly
        pred = torch.stack(pred_s).permute(1, 0, 2, 3, 4).contiguous()
        return pred

    def _get_msg(self, x1, x2):
        b, c, h, w = x1.shape
        wgt = self.C_wgt(x1 - x2).unsqueeze(1).repeat(1, c//(self.n_node-1), 1, 1, 1).view(b, c, h, w)    #  edge---wly
        out = x2 * torch.sigmoid(wgt)    # m--------------------wly
        return out
