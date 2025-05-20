import numpy as np
import cv2
import os
from skimage.io import imsave
# from evaluation_wly import Evaluator
from util.Evaluator import Evaluator


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb' or mode == 'Y' or mode == 'Cr' or mode == 'Cb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        return img
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
        return img
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        img = cv2.split(img)
        return img
    elif mode == 'Y':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(img)
        return Y
    elif mode == 'Cr':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(img)
        return Cr
    elif mode == 'Cb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(img)
        return Cb


def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image)


if __name__ == "__main__":
    ori_img_folder = './test_img/TNO'
    test_out_folder = './test_result/TNO'


    eval_folder = test_out_folder
    ori_img_folder = ori_img_folder
    data_range = 255
    metric_result = np.zeros((15))
    for img_name in os.listdir(os.path.join(ori_img_folder, "ir")):
        ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), 'GRAY')
        vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), 'GRAY')

        fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0] + ".png"), 'GRAY')
        # print(vi,fi)
        # print(ir.shape)
        metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                      , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                      , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                      , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi, data_range)
                                      , Evaluator.MSE(fi, ir, vi), Evaluator.CC(fi, ir, vi)
                                      , Evaluator.PSNR(fi, ir, vi), Evaluator.AG1(fi)
                                      , Evaluator.Nabf(fi, ir, vi), Evaluator.MS_SSIM(fi, ir, vi)
                                      , Evaluator.LPIPS(fi, ir, vi)
                                   ])

    metric_result /= len(os.listdir(eval_folder))
    print(
        "\t\t\t EN\t\t SD\t\t SF\t\t MI\t\t SCD\t VIF\t Qabf\t SSIM\t MSE\t CC\t\t PSNR\t AG\t\tNabf\t\tMS_SSIM\t\tLPIPS")
    print("MayNet" + '\t' + str(np.round(metric_result[0], 4)) + '\t'
          + str(np.round(metric_result[1], 4)) + '\t'
          + str(np.round(metric_result[2], 4)) + '\t'
          + str(np.round(metric_result[3], 4)) + '\t'
          + str(np.round(metric_result[4], 4)) + '\t'
          + str(np.round(metric_result[5], 4)) + '\t'
          + str(np.round(metric_result[6], 4)) + '\t'
          + str(np.round(metric_result[7], 4)) + '\t'
          + str(np.round(metric_result[8], 4)) + '\t'
          + str(np.round(metric_result[9], 4)) + '\t'
          + str(np.round(metric_result[10], 4)) + '\t'
          + str(np.round(metric_result[11], 4)) + '\t'
          + str(np.round(metric_result[12], 4)) + '\t'
          + str(np.round(metric_result[13], 4)) + '\t'
          + str(np.round(metric_result[14], 4)) + '\t'
          )
    print("=" * 80)
