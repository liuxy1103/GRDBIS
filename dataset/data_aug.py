import cv2
import torch
import random
import numpy as np
from dataset.transforms import random_crop
from scipy.ndimage.interpolation import zoom

def resize_(img, height, width):
    '''
    Resize a 3D array (image) to the size specified in parameters
    '''
    zoom_h = float(height) / img.shape[0]
    zoom_w = float(width) / img.shape[1]
    img = zoom(img, [zoom_h, zoom_w, 1], mode='nearest', order=0)
    return img

def scale(img,ins,seg):
    h = img.size(1)
    w = img.size(2)

    # have masks be of shape (1,h,w)
    seg = np.expand_dims(seg, axis=-1)
    ins = np.expand_dims(ins, axis=-1)
    ins = resize_(ins, h, w)
    seg = resize_(seg, h, w)
    seg = seg.squeeze()
    ins = ins.squeeze()

    return ins, seg

def flip_crop(img, ins, seg, flip=True, crop=True, imsize=256):
    h = img.size(1)
    w = img.size(2)
    seg = np.expand_dims(seg, axis=0)
    ins = np.expand_dims(ins, axis=0)

    if random.random() < 0.5 and flip:
        img = np.flip(img.numpy(),axis=2).copy()
        img = torch.from_numpy(img)
        ins = np.flip(ins,axis=2).copy()
        seg = np.flip(seg,axis=2).copy()

    ins = torch.from_numpy(ins)
    seg = torch.from_numpy(seg)
    if crop:
        img, ins, seg = random_crop([img,ins,seg],(imsize,imsize), (h,w))
    return img, ins, seg

def aug_flip(data, label):
    rule = np.random.randint(2, size=3)
    # x reflection
    if rule[0]:
        data = data[::-1, :, :]
        label = label[::-1, :]
    # y reflection
    if rule[1]:
        data = data[:, ::-1, :]
        label = label[:, ::-1]
    # transpose in xy
    if rule[2]:
        data = np.transpose(data, (1,0,2))
        label = np.transpose(label, (1,0))
    return data, label

def aug_crop(data, label, size=544, scale_min=0.7, scale_max=1.2):
    size_h, size_w = label.shape
    if random.random() > 0.5:
        scale_h = np.random.uniform(scale_min, scale_max)
        scale_w = np.random.uniform(scale_min, scale_max)
    else:
        scale_h = 1.0
        scale_w = 1.0
    out_size_h = int(size * scale_h)
    if out_size_h > size:
        out_size_h = size
    out_size_w = int(size * scale_w)
    if out_size_w > size:
        out_size_w = size
    start_point_h = np.random.randint(0, size_h-out_size_h+1)
    start_point_w = np.random.randint(0, size_w-out_size_w+1)
    out_data = data[start_point_h:start_point_h+out_size_h, start_point_w:start_point_w+out_size_w, :]
    out_label = label[start_point_h:start_point_h+out_size_h, start_point_w:start_point_w+out_size_w]
    if out_label.shape[0] != size or out_label.shape[1] != size:
        out_data = cv2.resize(out_data, (size, size), interpolation=cv2.INTER_LINEAR)
        out_label = cv2.resize(out_label, (size, size), interpolation=cv2.INTER_NEAREST)
    return out_data, out_label