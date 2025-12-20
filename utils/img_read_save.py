import numpy as np
import cv2
import os
from skimage.io import imsave
#
# def image_read_cv2(path, mode='RGB'):
#     img_BGR = cv2.imread(path).astype('float32')
#     assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
#     if mode == 'RGB':
#         img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
#     elif mode == 'GRAY':
#         img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
#     elif mode == 'YCrCb':
#         img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
#     return img

def image_read_cv2(path, mode='RGB'):
    print(path)
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