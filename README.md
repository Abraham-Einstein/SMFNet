# A Semantic-Aware and Multi-Guided Network for Infrared-Visible Image Fusion
[Xiaoli Zhang](https://zhangxiaolijlu.github.io/), [Liying Wang](https://blog.csdn.net/weixin_46202235), [Libo Zhao](), [Xiongfei Li]() and [Siwei Ma](https://idm.pku.edu.cn/en/info/1009/1017.htm)

-[*[ArXiv]*](https://arxiv.org/abs/2407.06159)
## 🌟 Update
- [2024/6] The original manuscript was uploaded to arXiv.
- [2024/8] The manuscript has been submitted to IEEE Transactions on Multimedia.
- [2025/5] We have received the acceptance notification😊

## 📚 Abstract

### 🚀 Training
**1. Virtual Environment**
```
# create virtual environment
conda create -n cddfuse python=3.8.0
conda activate SMFNet
# select the pytorch-gpu version yourself
# install SMFNet requirements
pip install -r requirements.txt
```
**2. Training Dataset**

Download the MSRS dataset from [this link](https://github.com/Linfeng-Tang/MSRS) and place it inside your main project folder in the ``'./MSRS_train/'``.

**3. Pre-Processing**

Run 
```
python dataprocessing.py
``` 
and the processed training dataset is in ``'./data/MSRS_train_imgsize_128_stride_200.h5'``. This processing way is the same as CDDFuse.

**4. Training**

Run 
```
python train.py
``` 
and the trained model is available in ``'./models/'``.

### Testing

**1. Pretrained models**

Pretrained models are available in ``'./models/SMFNet_IVF.pth'`` and ``'./models/SMFNet_MIF.pth'``, which are responsible for the Infrared-Visible Fusion (IVF) and Medical Image Fusion (MIF) tasks, respectively. 

**2. Test datasets**

The test datasets includes ``'./test_img/RoadScene'``, ``'./test_img/TNO'`` for IVF, ``'./test_img/MRI-CT'``, ``'./test_img/MRI-PET'`` and ``'./test_img/MRI-SPECT'`` for MIF. Due to the limited number of pages, we only show MRI-PET fusion results in the paper. 


**3. Results in Our Paper**

If you want to use our SMFNet and obtain the fusion results in our paper, please run 
```
python test.py
```
which can match the results in our original paper.

## 📝 Citation

```
@article{zhang2024semantic,
  title={A Semantic-Aware and Multi-Guided Network for Infrared-Visible Image Fusion},
  author={Zhang, Xiaoli and Wang, Liying and Zhao, Libo and Li, Xiongfei and Ma, Siwei},
  journal={arXiv preprint arXiv:2407.06159},
  year={2024}
}
```
## 😊 Any question

If you have any question, please feel free to contact with [Liying Wang](https://blog.csdn.net/weixin_46202235) at `my_lnnu@163.com` or `liyingw23@gmails.jlu.edu.cn`.


## 💡Acknowledgements

The codes are based on [CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse), [DeepMiH](https://github.com/TomTomTommi/DeepMIH), [IGNet](https://github.com/lok-18/IGNet), and [MGDN](https://github.com/Guanys-dar/MGDN). Please also follow their licenses. Thanks for their awesome work. Additionally, if you are attracted by [PromptFusion](https://github.com/hey-it-s-me/PromptFusion), please also refer to their papers.

