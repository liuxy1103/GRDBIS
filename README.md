# GRDBIS: Graph Relation Distillation for Efficient Biomedical Instance Segmentation 
**Under Review**
[Arxiv](https://arxiv.org/abs/2401.06370)



This repo contains the code of our paper submitted, which is an extension version of our work "Efficient Biomedical Instance Segmentation via Knowledge Distillation" published in MICCAI-22.


## Installaion
This code was implemented with Pytorch 1.0.1 (later versions may work), CUDA 9.0, Python 3.7.4. 

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
The related [pretrained models](https://drive.google.com/drive/folders/1AvPbzRxQJABvvyraoFrElxFtcXlHZJS0) are available, please refer to the testing command for evaluation.

## Workflow
![image](https://user-images.githubusercontent.com/54794058/233539448-3b417dba-2951-4668-bf33-f55a789733e8.png)


## Dataset

| Datasets                                                     | Training set | Validation set | Test set      | Download (Processed)                                         |
| ------------------------------------------------------------ | ------------ | -------------- | ------------- | ------------------------------------------------------------ |
| [CVPPP (A1)](https://competitions.codalab.org/competitions/18405) | 530x500x108  | 530x500x20     | 530x500x33    | [BaiduYun](https://pan.baidu.com/s/1fH5ek1Zy5pz5R0HQfaUbTg) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1IsPmaBjDXkSyzPXKjB4GIwHb_5pVVXBe?usp=sharing) |
| [BBBC039V1](https://bbbc.broadinstitute.org/BBBC039)         | 520x696x100  | 520x696x50     | 520x696x50    | [BaiduYun](https://pan.baidu.com/s/1S2tYjfN4-mMIRgnxfY8QsQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1IsPmaBjDXkSyzPXKjB4GIwHb_5pVVXBe?usp=sharing) |
| [AC3/AC4](https://software.rc.fas.harvard.edu/lichtman/vast/<br/>AC3AC4Package.zip) | 1024x1024x80 | 1024x1024x20   | 1024x1024x100 | [BaiduYun](https://pan.baidu.com/s/1rY6MlALpzvkYTgn04qghjQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1IsPmaBjDXkSyzPXKjB4GIwHb_5pVVXBe?usp=sharing) |

Download and unzip them in corresponding folders in './data'.


## Contact
If you have any problem with the released code, please contact me by email (liuxyu@mail.ustc.edu.cn).
