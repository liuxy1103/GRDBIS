import cv2
import torch
import random
import numpy as np
import torch.nn.functional as F

def tensor2img(img):
    img = img.numpy()
    std = np.asarray([0.229, 0.224, 0.225])
    std = std[:, np.newaxis, np.newaxis]
    mean = np.asarray([0.485, 0.456, 0.406])
    mean = mean[:, np.newaxis, np.newaxis]
    img = img * std + mean
    return img

def img2tensor(img):
    std = np.asarray([0.229, 0.224, 0.225])
    std = std[:, np.newaxis, np.newaxis]
    mean = np.asarray([0.485, 0.456, 0.406])
    mean = mean[:, np.newaxis, np.newaxis]
    img = (img.astype(np.float32) - mean) / std
    return torch.from_numpy(img.astype(np.float32))

def add_gauss_noise(imgs, min_std=0, max_std=0.05, norm_mode='trunc'):
    if min_std == max_std:
        std = min_std
    else:
        std = random.uniform(min_std, max_std)
    gaussian = np.random.normal(0, std, (imgs[0].shape))
    imgs[0,...] = imgs[0,...] + gaussian
    imgs[1,...] = imgs[1,...] + gaussian
    imgs[2,...] = imgs[2,...] + gaussian
    if norm_mode == 'norm':
        imgs = (imgs-np.min(imgs)) / (np.max(imgs)-np.min(imgs))
    elif norm_mode == 'trunc':
        imgs = np.clip(imgs, 0, 1)
    else:
        raise NotImplementedError
    return imgs


def add_gauss_blur(imgs, min_kernel_size=1, max_kernel_size=7, min_sigma=0, max_sigma=1):
    outs = []
    kernel_size = random.randint(min_kernel_size // 2, max_kernel_size // 2)
    kernel_size = kernel_size * 2 + 1
    sigma = random.uniform(min_sigma, max_sigma)
    for k in range(imgs.shape[0]):
        temp = imgs[k]
        temp = cv2.GaussianBlur(temp, (kernel_size,kernel_size), sigma)
        outs.append(temp)
    outs = np.asarray(outs, dtype=np.float32)
    outs = np.clip(outs, 0, 1)
    return outs


def add_intensity(imgs, contrast_factor=0.1, brightness_factor=0.1):
    imgs *= 1 + (np.random.rand() - 0.5) * contrast_factor
    imgs += (np.random.rand() - 0.5) * brightness_factor
    imgs = np.clip(imgs, 0, 1)
    # imgs **= 2.0**(np.random.rand()*2 - 1)
    # imgs *= 1 + contrast_factor
    # imgs += brightness_factor
    # imgs = np.clip(imgs, 0, 1)
    return imgs


def corner_point(label_mask):
    xx, yy = np.where(label_mask == 1)
    xx_min = xx.min()
    xx_max = xx.max()
    yy_min = yy.min()
    yy_max = yy.max()
    return xx_min, xx_max, yy_min, yy_max

def add_mask(imgs, label_mask, min_mask_counts=0, max_mask_counts=20, min_mask_size=0, max_mask_size=20):
    mask = np.ones_like(imgs[0], dtype=np.float32)
    mask_counts = random.randint(min_mask_counts, max_mask_counts)
    mask_size_xy = random.randint(min_mask_size, max_mask_size)
    xx_min, xx_max, yy_min, yy_max = corner_point(label_mask)
    for k in range(mask_counts):
        my = random.randint(xx_min, xx_max-mask_size_xy)
        mx = random.randint(yy_min, yy_max-mask_size_xy)
        mask[my:my+mask_size_xy, mx:mx+mask_size_xy] = 0
    mean1 = np.sum(imgs[0] * label_mask) / np.sum(label_mask)
    mean2 = np.sum(imgs[1] * label_mask) / np.sum(label_mask)
    mean3 = np.sum(imgs[2] * label_mask) / np.sum(label_mask)
    imgs[0] = imgs[0] * mask + (1-mask) * mean1
    imgs[1] = imgs[1] * mask + (1-mask) * mean2
    imgs[2] = imgs[2] * mask + (1-mask) * mean3
    return imgs
