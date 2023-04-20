# GIDBIS
**TMI Under Review**

This repo contains code of our paper submmited to IEEE Transactions on Medical Imaging, which is an extension version of our work "Efficient Biomedical Instance Segmentation via Knowledge Distillation" published in MICCAI-22.


## Installaion
This code was implemented with Pytorch 1.0.1 (later versions may work), CUDA 9.0, Python 3.7.4 and Ubuntu 16.04. 

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows:

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/em_seg/v54_higra
```

## Training
###  2D dataset, take the C_elegans dataset as an example

```shell
 python trainKD_Celegans.py -c=seg_snemi3d_d5_1024_u200 
```

###  3D dataset, take the AC3/4 dataset as an example

```shell
 python trainKD_EM.py -c=seg_snemi3d_d5_1024_u200 
```

## Pretrained Model
The related [pretrained models](https://drive.google.com/drive/folders/1AvPbzRxQJABvvyraoFrElxFtcXlHZJS0) are available, please refer to the testing command for evaluating.


## Testing


###  2D dataset, take the C_elegans dataset as an example

```shell
 python inference_Celegans.py 
```

###  3D dataset, take the AC3/4 dataset as an example

```shell
 python inference_EM.py --c=seg_ac34_dics_mala_o3_emb_aff_emb_KD_s3_div_multiple_aff10_node0.1_edge0.1_CIaff1_CInode0_CIedge1 --mn=2023-02-27--01-38-52_seg_ac34_dics_mala_o3_emb_aff_emb_KD_s3_div_multiple_aff10_node0.1_edge0.1_CIaff1_CInode0_CIedge1 -m=ac3
```








## Contact
If you have any problem with the released code, please contact me by email (liuxyu@mail.ustc.edu.cn).
