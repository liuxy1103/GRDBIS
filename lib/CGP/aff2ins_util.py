# import scipy.stats
import numpy as np
from PIL import Image
import time
import functools
import cv2

def Downscale(map, stride=1):
    stride = int(stride)
    start = stride//2
    if len(map.shape) == 1:
        return map[start::stride]
    elif len(map.shape) == 2:
        return map[start::stride, start::stride]
    elif len(map.shape) >= 3:
        return map[:, start::stride, start::stride]
    else:
        raise ValueError('map.shape={}'.format(map.shape))

def entropy(x,y):
    assert len(x.shape)==len(y.shape)
    if len(x.shape)==1:
        return (x * np.log2(x / y)).sum()
    else:
        assert len(x.shape)==2
        return (x * np.log2(x / y)).sum(axis=1)

def kl_similar(x, y):
    eps = 1e-12
    x = np.asarray(x)+eps
    y = np.asarray(y)+eps
    dist = 0.5 * entropy(x, y) + 0.5 * entropy(y, x)
    return np.exp(-dist)

def js_similar(x, y):
    eps = 1e-12
    x = np.asarray(x)+eps
    y = np.asarray(y)+eps
    m = (x+y)/2
    dist = 0.5 * entropy(x, m) + 0.5 * entropy(y, m)
    return np.exp(-dist)

def sig_trans(x, alpha=5):
    return 2 * (1 / (1 + np.exp(- alpha * x)) - 0.5)

def gasp_similar(x,y,alpha=5):
    eps = 1e-12
    x = np.asarray(x)
    y = np.asarray(y)

    x[x < eps]     = eps
    x[x > 1 - eps] = 1 - eps
    y[y < eps]     = eps
    y[y > 1 - eps] = 1 - eps

    return sig_trans((x * y).sum(axis=1), alpha)

def cubic(a,b,c,d,y,x):
    return (1-x)*(1-y)*a + (1-x)*y*b + x*(1-y)*c + x*y*d

def logit_transform(p, bias=0.5):
    p = np.asarray(p, dtype=np.float64)
    pp = np.minimum(np.maximum(p, 1e-12), 1.0 - 1e-12)
    return np.log((1. - pp) / pp) - np.log((1. - bias) / bias)

def dense_ind(index_map,stay_shape=False):
    mapping = np.unique(index_map).astype('int32')
    map_key = np.arange(len(mapping)).astype('int32')
    index = np.digitize(index_map.ravel(), mapping, right=True)
    if stay_shape:
        index_map = map_key[index].reshape(index_map.shape)
    else:
        index_map = map_key[index]
    return index_map

def for_view(ins_r):
    idxxx = np.random.permutation(len(np.unique(ins_r)))
    ins2id={ins:id for ins,id in zip(np.unique(ins_r),idxxx)}
    return np.vectorize(ins2id.get)(ins_r)

def nonoverlap_aff(aff_size, overlap):
    fullmask = np.array(range(aff_size**2)).reshape([aff_size, aff_size])
    begin = int((aff_size - 1) / 2 - overlap)
    end   = int((aff_size - 1) / 2 + overlap)
    overlapmask = fullmask[begin:end + 1, begin:end+1].reshape(-1).tolist()
    nonoverlap_aff = list(set(range(aff_size**2)).difference(set(overlapmask)))
    return nonoverlap_aff

def get_aff(label, aff_size, overlap=0):
    if len(label.shape) == 2:
        h,w=label.shape
        label=label.reshape([1 ,h, w])
    elif len(label.shape) == 3:
        h, w, c = label.shape
        label = label.reshape([1, h, w, c])
    ins_label=[]
    pad_size = int((aff_size - 1) / 2)
    for i in range(pad_size, -pad_size-1, -1):
        ins_label_h_shift = np.roll(label, i, axis=1)
        if i > 0:
            ins_label_h_shift[:, :i, :] = -1
        elif i < 0:
            ins_label_h_shift[:, i:, :] = -1
        for j in range(pad_size, -pad_size-1, -1):
            if abs(i)<=overlap and abs(j)<=overlap:
                continue
            ins_label_shift = np.roll(ins_label_h_shift, j, axis=2)
            if j > 0:
                ins_label_shift[:, :, :j] = -1
            elif j < 0:
                ins_label_shift[:, :, j:] = -1
            
            ins_label.append(ins_label_shift)
    return np.concatenate(ins_label, axis=0)

if __name__ == "__main__":
    label = np.random.randint(0,100, [5, 5])
    print(np.unique(label))
    print(np.unique(dense_ind(label)))

