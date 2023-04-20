import numpy as np
from elf.segmentation.mutex_watershed import mutex_watershed

def seg_mutex(affs, offsets=[[-1,0],[0,-1]], strides=[1,1], randomize_strides=False, mask=None):
    return mutex_watershed(1.0 - affs, offsets, strides, randomize_strides=randomize_strides, mask=mask)