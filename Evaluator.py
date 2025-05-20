
import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel, filters
from scipy.ndimage import convolve
import torch
import lpips


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


class Evaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None):
        if imgA is None:
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def AG1(cls, img):  # Average gradient
        cls.input_check(img)

        # Compute the gradient in both x and y directions
        Gx, Gy = np.gradient(img)

        # Compute the magnitude of the gradient at each pixel
        magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

        # Compute the average gradient
        return np.mean(magnitude)

    # import numpy as np
    # from scipy.ndimage import gradient
    @classmethod
    def AG_evaluation(cls, img):
        cls.input_check(img)
        if img.ndim == 3:
            # 将图像转换为double格式
            img = img.astype(float)

            # 获取图像的大小
            r, c, b = img.shape

            # 设置用于梯度计算的步长
            dx = 1
            dy = 1

            # 初始化一个数组以存储每个颜色通道的平均梯度
            g = np.zeros(b)

            # 循环遍历每个颜色通道
            for k in range(b):
                # 提取第k个颜色通道
                band = img[:, :, k]

                # 计算x和y方向上的梯度
                dzdx, dzdy = np.gradient(band, dx, dy)

                # 计算每个像素的梯度幅值
                s = np.sqrt((dzdx ** 2 + dzdy ** 2) / 2)

                # 计算当前颜色通道的平均梯度
                g[k] = np.sum(s) / ((r - 1) * (c - 1))

            # 计算所有颜色通道上平均梯度的均值
            outval = np.mean(g)

            return outval

        else:
            raise ValueError('输入参数数量错误！')

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return skm.mutual_info_score(image_F.flatten(), image_A.flatten()) + skm.mutual_info_score(image_F.flatten(),
                                                                                                   image_B.flatten())

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  # MSE
        cls.input_check(image_F, image_A, image_B)
        return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * np.log10(np.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    @classmethod
    def SCD(cls, image_F, image_A, image_B): # The sum of the correlations of differences
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
        return corr1 + corr2

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F)+cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls,ref, dist): # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        # 计算QABF
        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno

    @classmethod
    def Qabf_getArray(cls,img):
        # Sobel Operator Sobel
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0]=np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls,aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF,AAF,QgAF,QaAF,QAF = np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA)
        GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF* QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B, data_range):
        cls.input_check(image_F, image_A, image_B)
        return ssim(image_F,image_A, data_range=data_range)+ssim(image_F,image_B, data_range=data_range)

    @classmethod
    def Nabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        # Parameters for Petrovic Metrics Computation
        Td = 2
        wt_min = 0.001
        P = 1
        Lg = 1.5
        Nrg = 0.9999
        kg = 19
        sigmag = 0.5
        Nra = 0.9995
        ka = 22
        sigmaa = 0.5

        xrcw = image_F
        x1 = image_A
        x2 = image_B

        # Edge Strength & Orientation
        gvA, ghA = sobel(x1, axis=0), sobel(x1, axis=1)
        gA = np.sqrt(ghA ** 2 + gvA ** 2)

        gvB, ghB = sobel(x2, axis=0), sobel(x2, axis=1)
        gB = np.sqrt(ghB ** 2 + gvB ** 2)

        gvF, ghF = sobel(xrcw, axis=0), sobel(xrcw, axis=1)
        gF = np.sqrt(ghF ** 2 + gvF ** 2)

        # Relative Edge Strength & Orientation
        gAF = np.where(gA == 0, 0, np.minimum(gF, gA) / np.maximum(gF, gA))
        gBF = np.where(gB == 0, 0, np.minimum(gF, gB) / np.maximum(gF, gB))

        aA = np.arctan2(gvA, ghA)
        aB = np.arctan2(gvB, ghB)
        aF = np.arctan2(gvF, ghF)

        aAF = np.abs(np.abs(aA - aF) - np.pi / 2) * 2 / np.pi
        aBF = np.abs(np.abs(aB - aF) - np.pi / 2) * 2 / np.pi

        # Edge Preservation Coefficient
        QgAF = Nrg / (1 + np.exp(-kg * (gAF - sigmag)))
        QaAF = Nra / (1 + np.exp(-ka * (aAF - sigmaa)))
        QAF = np.sqrt(QgAF * QaAF)

        QgBF = Nrg / (1 + np.exp(-kg * (gBF - sigmag)))
        QaBF = Nra / (1 + np.exp(-ka * (aBF - sigmaa)))
        QBF = np.sqrt(QgBF * QaBF)

        # Total Fusion Performance (QABF)
        wtA = np.where(gA >= Td, wt_min * gA ** Lg, wt_min)
        wtB = np.where(gB >= Td, wt_min * gB ** Lg, wt_min)
        wt_sum = np.sum(wtA + wtB)

        QAF_wtsum = np.sum(QAF * wtA) / wt_sum
        QBF_wtsum = np.sum(QBF * wtB) / wt_sum
        QABF = QAF_wtsum + QBF_wtsum

        # Fusion Gain (QdeltaABF)
        Qdelta = np.abs(QAF - QBF)
        QCinfo = (QAF + QBF - Qdelta) / 2
        QdeltaAF = QAF - QCinfo
        QdeltaBF = QBF - QCinfo
        QdeltaAF_wtsum = np.sum(QdeltaAF * wtA) / wt_sum
        QdeltaBF_wtsum = np.sum(QdeltaBF * wtB) / wt_sum
        QdeltaABF = QdeltaAF_wtsum + QdeltaBF_wtsum

        QCinfo_wtsum = np.sum(QCinfo * (wtA + wtB)) / wt_sum
        QABF11 = QdeltaABF + QCinfo_wtsum

        # Fusion Loss (LABF)
        rr = np.where(gF <= gA, 1, 0) | np.where(gF <= gB, 1, 0)
        LABF = np.sum(rr * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum

        # Fusion Artifacts (NABF) by Petrovic
        na1 = np.where(gF > gA, 2 - QAF - QBF, 0)
        NABF1 = np.sum(na1 * (wtA + wtB)) / wt_sum

        # Fusion Artifacts (NABF) changed by B. K. Shreyamsha Kumar
        na = np.where(gF > gA, 1, 0)
        NABF = np.sum(na * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum

        return NABF

    @classmethod
    def MS_SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        # 计算MS-SSIM
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])  # 不同尺度的权重
        levels = weights.size

        mssim = np.zeros(levels)
        mcs = np.zeros(levels)
        mssim_1 = np.zeros(levels)
        mcs_1 = np.zeros(levels)

        img1 = image_F
        img2 = image_A
        img3 = image_B

        for i in range(levels):
            ssim_map, mcs_map = cls.ssim(img1, img3)
            mssim[i] = np.mean(ssim_map)
            mcs_map[i] = np.mean(mcs_map)

            img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
            img3 = cv2.resize(img2, (img3.shape[1] // 2, img3.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

        img1 = image_F
        for i in range(levels):
            ssim_map, mcs_map = cls.ssim(img1, img2)
            mssim_1[i] = np.mean(ssim_map)
            mcs_1[i] = np.mean(mcs_map)

            img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

        # 整体MS-SSIM计算
        overall_mssim = np.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1]) + np.prod(mcs_1[:-1] ** weights[:-1]) * (mssim_1[-1] ** weights[-1])

        return overall_mssim

    @classmethod
    def ssim(cls, img1, img2, k1=0.01, k2=0.03, win_size=11, L=255):
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2

        # print("img1 shape:", img1.shape)
        # print("img2 shape:", img2.shape)
        # 计算均值和方差
        mu1 = cv2.GaussianBlur(img1, (win_size, win_size), 1.5)
        # print("img1 shape:", img1.shape)
        mu2 = cv2.GaussianBlur(img2, (win_size, win_size), 1.5)
        # print("img2 shape:", img2.shape)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 * img1, (win_size, win_size), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, (win_size, win_size), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (win_size, win_size), 1.5) - mu1_mu2

        # 计算相似性度量
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

        return ssim_map, cs_map
    # @classmethod
    # def gaussian_kernel(cls, size, sigma):
    #     kernel = np.zeros((size, size))
    #     center = size // 2
    #     for i in range(size):
    #         for j in range(size):
    #             x = i - center
    #             y = j - center
    #             kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    #     return kernel / np.sum(kernel)
    # @classmethod
    # def ms_ssim(cls, img1, img2, max_val=255, level=5, weight=None, win_size=11, sigma=1.5):
    #
    #     img1 = img1.astype(np.float64)
    #     img2 = img2.astype(np.float64)
    #     img1_sq = img1 ** 2
    #     img2_sq = img2 ** 2
    #     img12 = img1 * img2
    #
    #     weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]) if weight is None else weight
    #     levels = min(level, int(np.floor(np.log2(min(img1.shape[:2])) - 3)))
    #     scales = np.array([cv2.resize(cls.gaussian_kernel(win_size, sigma), img1.shape[:2][::-1])])
    #     for i in range(1, levels):
    #         prev_size = scales[-1].shape[:2]
    #         size = (int(np.ceil(prev_size[0] / 2)), int(np.ceil(prev_size[1] / 2)))
    #         if size[0] < 2 or size[1] < 2:
    #             break
    #         kernel = cls.gaussian_kernel(win_size, sigma)
    #         kernel = cv2.resize(kernel, size[::-1])
    #         kernel /= np.sum(kernel)
    #         scales = np.append(scales, [kernel], axis=0)
    #
    #     mssim = np.array([])
    #     mcs = np.array([])
    #     for i in range(levels):
    #         ssim_map = np.array([])
    #         cs_map = np.array([])
    #         for j in range(weights.size):
    #             if i == 0 and j == 0:
    #                 ssim_map = np.ones(img1.shape[:2])
    #                 cs_map = np.ones(img1.shape[:2])
    #             else:
    #                 img1 = cv2.filter2D(img1, -1, scales[i], borderType=cv2.BORDER_REFLECT)
    #                 img2 = cv2.filter2D(img2, -1, scales[i], borderType=cv2.BORDER_REFLECT)
    #                 img1_sq = cv2.filter2D(img1_sq, -1, scales[i], borderType=cv2.BORDER_REFLECT)
    #                 img2_sq = cv2.filter2D(img2_sq, -1, scales[i], borderType=cv2.BORDER_REFLECT)
    #                 img12 = cv2.filter2D(img12, -1, scales[i], borderType=cv2.BORDER_REFLECT)
    #
    #                 mu1 = img1
    #                 mu2 = img2
    #                 mu1_sq = mu1 ** 2
    #                 mu2_sq = mu2 ** 2
    #                 mu12 = mu1 * mu2
    #                 sigma1_sq = img1_sq - mu1_sq
    #                 sigma2_sq = img2_sq - mu2_sq
    #                 sigma12 = img12 - mu12
    #
    #                 ssim_map_ = ((2 * mu12 + max_val ** 2) * (2 * sigma12 + max_val ** 2)) / (
    #                             (mu1_sq + mu2_sq + max_val ** 2) * (sigma1_sq + sigma2_sq + max_val ** 2))
    #                 cs_map_ = (2 * sigma12 + max_val ** 2) / (sigma1_sq + sigma2_sq + max_val ** 2)
    #
    #                 if weights[j] != 1:
    #                     ssim_map_ = ssim_map_ ** weights[j]
    #                     cs_map_ = cs_map_ ** weights[j]
    #
    #                 ssim_map = np.append(ssim_map, [np.mean(ssim_map_)])
    #                 cs_map = np.append(cs_map, [np.mean(cs_map_)])
    #
    #         mssim = np.append(mssim, [np.prod(ssim_map ** weights)])
    #         mcs = np.append(mcs, [np.prod(cs_map ** weights)])
    #
    #     return np.prod(mcs[:levels - 1] ** weights[:levels - 1]) * (mssim[levels - 1] ** weights[levels - 1])
    #
    # @classmethod
    # def MS_SSIM(cls, image_F, image_A, image_B):
    #     cls.input_check(image_F, image_A, image_B)
    #     return cls.ms_ssim(image_F, image_A) + cls.ms_ssim(image_F, image_B)



    @classmethod
    def LPIPS(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        # 加载预训练的LPIPS模型
        image1 = image_F
        image2 = image_A
        image3 = image_B
        lpips_model = lpips.LPIPS(net="alex")

        # 将图像转换为PyTorch的Tensor格式
        image1_tensor = torch.tensor(np.array(image1)).unsqueeze(0).unsqueeze(0).float() / 255.0
        image2_tensor = torch.tensor(np.array(image2)).unsqueeze(0).unsqueeze(0).float() / 255.0
        image3_tensor = torch.tensor(np.array(image3)).unsqueeze(0).unsqueeze(0).float() / 255.0

        # 使用LPIPS模型计算距离
        distance1 = lpips_model(image1_tensor, image2_tensor)
        distance2 = lpips_model(image1_tensor, image3_tensor)


        return (distance1.item() + distance2.item()) / 2


def VIFF(image_F, image_A, image_B):
    refA=image_A
    refB=image_B
    dist=image_F

    sigma_nsq = 2
    eps = 1e-10
    numA = 0.0
    denA = 0.0
    numB = 0.0
    denB = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        # Create a Gaussian kernel as MATLAB's
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode='valid')
            refB = convolve2d(refB, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            refA = refA[::2, ::2]
            refB = refB[::2, ::2]
            dist = dist[::2, ::2]

        mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
        mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq_A = mu1A * mu1A
        mu1_sq_B = mu1B * mu1B
        mu2_sq = mu2 * mu2
        mu1A_mu2 = mu1A * mu2
        mu1B_mu2 = mu1B * mu2
        sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1_sq_A
        sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1_sq_B
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A_mu2
        sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B_mu2

        sigma1A_sq[sigma1A_sq < 0] = 0
        sigma1B_sq[sigma1B_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        gA = sigma12_A / (sigma1A_sq + eps)
        gB = sigma12_B / (sigma1B_sq + eps)
        sv_sq_A = sigma2_sq - gA * sigma12_A
        sv_sq_B = sigma2_sq - gB * sigma12_B

        gA[sigma1A_sq < eps] = 0
        gB[sigma1B_sq < eps] = 0
        sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
        sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
        sigma1A_sq[sigma1A_sq < eps] = 0
        sigma1B_sq[sigma1B_sq < eps] = 0

        gA[sigma2_sq < eps] = 0
        gB[sigma2_sq < eps] = 0
        sv_sq_A[sigma2_sq < eps] = 0
        sv_sq_B[sigma2_sq < eps] = 0

        sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
        sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
        gA[gA < 0] = 0
        gB[gB < 0] = 0
        sv_sq_A[sv_sq_A <= eps] = eps
        sv_sq_B[sv_sq_B <= eps] = eps

        numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
        numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
        denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
        denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

    vifpA = numA / denA
    vifpB =numB / denB

    if np.isnan(vifpA):
        vifpA=1
    if np.isnan(vifpB):
        vifpB = 1
    return vifpA+vifpB


