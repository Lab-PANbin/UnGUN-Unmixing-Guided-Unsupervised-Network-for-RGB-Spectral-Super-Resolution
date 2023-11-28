# UnGUN

This is a PyTorch implementation of ‘Unmixing Guided Unsupervised Network for RGB
Spectral Super-Resolution’.  Qiaoying Qu, Bin Pan, Xia Xu, Tao Li and Zhenwei Shi.


```shell
UNGUN_CODE
├── data # hyperspectral images for training/testing
├── endmember #initialization for decoder1
├── guidance_data # guidance hyperspectral images for training/testing
├── pretrained_model # pretrained models 
├── save # save path
├── SRF # spectral response function
├── layer.py
├── load_data.py
├── model.py
├── test.py 
└── train.py
```
data, guidance_data and save folders can be downloaded from the following link:  [LINK](https://pan.baidu.com/s/1scUKeK0Fh54ZY_-3yikhmw) 
code: qwer


*This implementation is for non-commercial research use only. If you find this code useful in your research, please cite the above paper.*

```latex
@ARTICLE{qu@ungun,
  author={Qu, Qiaoying and Pan, Bin and Xu, Xia and Li, Tao and Shi, Zhenwei},
  journal={IEEE Transactions on Image Processing}, 
  title={Unmixing Guided Unsupervised Network for RGB Spectral Super-Resolution}, 
  year={2023},
  volume={32},
  number={},
  pages={4856-4867},
  doi={10.1109/TIP.2023.3299197}}

```

