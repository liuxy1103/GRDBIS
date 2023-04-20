import os
import cv2
import numpy as np
from PIL import Image

def Affinity_generator_new(img_ins):
    """
    SSAP resolution 1/2, 1/4, 1/16, 1/32, 1/64 
    """
    img_size = 512
    aff_r = 5
    aff_resolution = 5
    # img_ins = Image.fromarray(img_ins)
    # 初始化一个aff_r * aff_r^2 * size * size
    aff_map = np.zeros((aff_resolution, aff_r**2, img_size, img_size))
    ins_width, ins_height = img_ins.shape[0], img_ins.shape[1]

    for mul in range(aff_resolution):
        #resize大小后的ins, resize后的图片大小,instance最近邻插值
        img_size = img_size // 2
        ins_downsampe = cv2.resize(img_ins, (img_size,img_size), cv2.INTER_NEAREST)
        # tree-ins_downsampe
        
        #按affinity kernel半径padding
        ins_pad = cv2.copyMakeBorder(ins_downsampe,int(aff_r),int(aff_r),int(aff_r),int(aff_r),cv2.BORDER_CONSTANT, value=(0,0,0))
        aff_compare = np.zeros((aff_r**2, img_size, img_size, 3))
        # 对25个affinity kernel上进行错位填充ins
        for i in range(aff_r):
            for j in range(aff_r):
                aff_compare[i*aff_r+j] = ins_pad[i:i+img_size, j:j+img_size]

        # 相同物体affinity=1 不同affinity=0
        aff_data = np.where((aff_compare[:, :, :, 0] == ins_downsampe[:, :, 0])
                            & (aff_compare[:, :, :, 1] == ins_downsampe[:, :, 1])
                            & (aff_compare[:, :, :, 2] == ins_downsampe[:, :, 2]), 1, 0)

        # aff_data = self.transform(aff_data.transpose(1, 2, 0))
        aff_map[mul, :, 0:img_size, 0:img_size] = aff_data
    return aff_map

if __name__ == '__main__':
    img = np.asarray(Image.open('./plant015_label.png'))
    print(img.max())
    print(img.shape)
    img_padding = np.zeros((512, 512), dtype=np.uint8)
    img_padding[:, 6:-6] = img[9:-9, :]
    print(img_padding.shape)
    img_padding_3 = img_padding[:,:,np.newaxis]
    img_padding_3 = np.repeat(img_padding_3, 3, 2)

    affs = Affinity_generator_new(img_padding_3)
    print(affs.shape)

    mask = np.zeros((2, 512, 512), dtype=np.uint8)
    mask[0] = img_padding == 0
    mask[1] = img_padding != 0
    # np.save('./mask.npy', mask)
    # np.save('./affs_map.npy', affs)

    # img1 = affs[1]
    # img_all = img1[0]
    # for i in range(24):
    #     img_all = np.concatenate([img_all, img1[i+1]], axis=0)
    # img_all = (img_all * 255).astype(np.uint8)
    # Image.fromarray(img_all).save('./img_all.png')
