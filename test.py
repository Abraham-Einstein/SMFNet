from net_final import Restormer_Encoder, BaseFeatureExtraction, Restormer_Decoder, DetailFeatureExtraction, \
    CGR_backbone
import os
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2
import warnings
import logging
import cv2

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

#
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ckpt_path = r"models/model_VIF.pth"   #  ---------stage--two----cc_loss_G------wly


for dataset_name in ["TNO"]:
    print("\n" * 2 + "=" * 80)
    model_name = "CDDFuse    "
    print("The MSRS_test result of " + dataset_name + ' :')
    test_folder = os.path.join('./test_img', dataset_name)
    test_out_folder = os.path.join('./test_result', dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction()).to(device)
    GraphModel = nn.DataParallel(CGR_backbone()).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    GraphModel.load_state_dict(torch.load(ckpt_path)['GraphModel'])

    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()
    GraphModel.eval()


    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, "ir")):
            # read image as YCrCb

            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='Y')[
                           np.newaxis, np.newaxis, ...] / 255.0
            VIS_Cr = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='Cr')[
                         np.newaxis, np.newaxis, ...] / 255.0
            VIS_Cb = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='Cb')[
                         np.newaxis, np.newaxis, ...] / 255.0

            data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, _, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, _, feature_I = Encoder(data_IR)
            feature_GCN, _, _ = GraphModel(feature_V, feature_I, flag=True)  # ---feature from restormer block---wly

            # ------------------------------------------------------- #

            feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)

            # -------------------------------------------------------

            data_Fuse, feature_F = Decoder(data_VIS, feature_F_D + feature_F_B + feature_GCN)

            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            data_Fuse = data_Fuse.cpu().numpy()
            ######################################################################3
            merged_YCrCb = np.concatenate((data_Fuse, VIS_Cr, VIS_Cb), axis=1)
            fi = np.squeeze((merged_YCrCb * 255))

            color_image = fi.transpose(1, 2, 0)

            color_image = cv2.cvtColor(color_image, cv2.COLOR_YCrCb2BGR)

            cv2.imwrite(os.path.join(test_out_folder, "{}.png".format(img_name.split(sep='.')[0])), color_image)






