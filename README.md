# A Semantic-Aware and Multi-Guided Network for Infrared-Visible Image Fusion
## Update
## Citation
## Abstract
### 🏊 Training
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
and the processed training dataset is in ``'./data/MSRS_train_imgsize_128_stride_200.h5'``. This process is the same as CDDFuse.
**4. CDDFuse Training**

Run 
```
python train.py
``` 
and the trained model is available in ``'./models/'``.

### 🏄 Testing

**1. Pretrained models**

Pretrained models are available in ``'./models/SMFNet_IVF.pth'`` and ``'./models/SMFNet_MIF.pth'``, which are responsible for the Infrared-Visible Fusion (IVF) and Medical Image Fusion (MIF) tasks, respectively. 

**2. Test datasets**

The test datasets includes ``'./test_img/RoadScene'``, ``'./test_img/TNO'`` for IVF, ``'./test_img/MRI-CT'``, ``'./test_img/MRI-PET'`` and ``'./test_img/MRI-SPECT'`` for MIF. Due to the limited number of pages, we only show MRI-PET fusion results in the paper. 


**3. Results in Our Paper**

If you want to infer with our SMFNet and obtain the fusion results in our paper, please run 
```
python test.py
```
which can match the results in our original paper.
