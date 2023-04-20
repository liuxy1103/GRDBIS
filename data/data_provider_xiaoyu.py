# -*- coding: utf-8 -*-
# @Time    : 2020/5/16 12:50
# @Author  : Bo Hu
# @Email   : hubosist@mail.ustc.edu.cn
# @Software: PyCharm

import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from collections import defaultdict
from skimage import io
from torchvision import transforms as tfs
import PIL.Image as Image
import cv2
from torchvision import transforms
import numpy
import random

class ToLogits(object):
    def __init__(self, expand_dim=None):
        self.expand_dim = expand_dim

    def __call__(self, pic):
        if pic.mode == 'I':
            img = torch.from_numpy(numpy.array(pic, numpy.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(numpy.array(pic, numpy.int32, copy=True))
        elif pic.mode == 'F':
            img = torch.from_numpy(numpy.array(pic, numpy.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(numpy.array(pic, numpy.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if self.expand_dim is not None:
            return img.unsqueeze(self.expand_dim)
        return img


class CVPPP(Dataset):
    def __init__(self, dir, mode, size, num_train=108, padding=True):
        self.size = size
        self.dir = dir
        self.mode = mode
        self.padding = padding
        if (self.mode != "train") and (self.mode != "validation") and (self.mode != "test"):
            raise ValueError("The value of dataset mode must be assigned to 'train' or 'validation'")
        if self.mode == "validation":
            self.dir = os.path.join(dir, "train")
        else:
            self.dir = os.path.join(dir, mode)
        # self.path_labels = os.path.join(dir,'leftImg8bit',mode)
        self.id_num = os.listdir(self.dir)  # all file
        self.id_img = [f for f in self.id_num if 'rgb' in f]
        self.id_label = [f for f in self.id_num if 'label' in f]
        self.id_fg = [f for f in self.id_num if 'fg' in f]

        self.id_img.sort(key=lambda x: int(x[5:8]))
        self.id_label.sort(key=lambda x: int(x[5:8]))
        self.id_fg.sort(key=lambda x: int(x[5:8]))

        if self.mode == "validation":
            self.id_img = self.id_img[num_train:]
            self.id_label = self.id_label[num_train:]
            self.id_fg = self.id_fg[num_train:]
        if self.mode == "train":
            self.id_img = self.id_img[:num_train]
            self.id_label = self.id_label[:num_train]
            self.id_fg = self.id_fg[:num_train]
        print('The number of %s image is %d' % (self.mode, len(self.id_img)))

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        self.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.RandomResizedCrop(self.size, scale=(0.7, 1.)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        self.target_transform_test = transforms.Compose(
             [ToLogits()])

        self.target_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.RandomResizedCrop(self.size, scale=(0.7, 1.), interpolation=0),
             ToLogits()])

    def __len__(self):
        return len(self.id_img)

    def __getitem__(self, id):
        data = Image.open(os.path.join(self.dir, self.id_img[id])).convert('RGB')
        if self.mode != 'test':
            label = Image.open(os.path.join(self.dir, self.id_label[id]))
        else:
            label = Image.open(os.path.join(self.dir, self.id_fg[id]))

        if self.padding:
            data = np.asarray(data)
            data = np.pad(data, ((7,7),(22,22),(0,0)), mode='reflect')
            data = Image.fromarray(data)
            label = np.asarray(label)
            label = np.pad(label, ((7,7),(22,22)), mode='constant')
            label = Image.fromarray(label)

        if self.mode == 'train':
            seed = np.random.randint(2147483647)
            random.seed(seed)
            data = self.transform(data)
            random.seed(seed)
            label = self.target_transform(label)
        else:
            data = self.transform_test(data)
            label = self.target_transform_test(label)
        return data, label


def draw_fragments_2d(pred):
    m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    # print("the number of instance is %d" % size)
    color_pred = np.zeros([m, n, 3], dtype=np.uint8)
    idx = np.searchsorted(ids, pred)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,i] = color_val[idx]
    return color_pred


if __name__ == '__main__':
    data_dir = r'D:\expriments\affinity_CVPPP\data\A1'
    train_img_height = 544
    output_path = '../data_temp'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    trainset = CVPPP(dir=data_dir,mode="test",size=train_img_height, num_train=108)

    std = np.asarray([0.229, 0.224, 0.225])
    std = std[np.newaxis, np.newaxis, :]
    mean = np.asarray([0.485, 0.456, 0.406])
    mean = mean[np.newaxis, np.newaxis, :]
    for i, (data, label, fg) in enumerate(trainset):
        # data, label, fg = iter(trainset).__next__()
        # print(data.shape, label.shape, fg.shape)
        data = data.numpy()
        data = np.transpose(data, (1,2,0))
        data = ((data * std + mean) * 255).astype(np.uint8)

        label = np.squeeze(label.numpy().astype(np.uint8))
        lb_color = draw_fragments_2d(label)

        fg = np.squeeze((fg.numpy() * 255).astype(np.uint8))
        fg = fg[:, :, np.newaxis]
        fg = np.repeat(fg, 3, 2)

        im_cat = np.concatenate([data, lb_color, fg], axis=1)
        Image.fromarray(im_cat).save(os.path.join(output_path, str(i).zfill(4)+'.png'))
