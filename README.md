# GIDBIS
**MIA Under Review**

This repo contains code of our paper submmited to IEEE Transactions on Medical Imaging, which is an extension version of our work "Efficient Biomedical Instance Segmentation via Knowledge Distillation" published in MICCAI-22.


## Installaion
This code was implemented with Pytorch 1.0.1 (later versions may work), CUDA 9.0, Python 3.7.4 and Ubuntu 16.04. 

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows:

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/em_seg/v54_higra
```

## Training
###  2D dataset, take the network MobileNetV2 on the C_elegans dataset as an example

```shell
 python trainKD_Celegans.py -c=Celegans_ResUNet_MobileNetV2
```

###  3D dataset, take the network MALA-tiny on the AC3/4 dataset as an example

```shell
 python trainKD_EM.py -c=AC34_MALA_MALA-tiny
```


## Testing


###  2D dataset, take the network MobileNetV2 on the C_elegans dataset as an example

```shell
 python inference_Celegans.py -mn=ResUNet_MobileNetV2_KD
```

###  3D dataset, take the network MALA-tiny on the AC3/4 dataset as an example

```shell
 python inference_EM.py -mn=MALA-tiny_KD
```

## Pretrained Model
The related [pretrained models](https://drive.google.com/drive/folders/1AvPbzRxQJABvvyraoFrElxFtcXlHZJS0) are available, please refer to the testing command for evaluating.

## Workflow
![image](https://user-images.githubusercontent.com/54794058/233539448-3b417dba-2951-4668-bf33-f55a789733e8.png)





## Contact
If you have any problem with the released code, please contact me by email (liuxyu@mail.ustc.edu.cn).
