import os
import cv2
import h5py
import numpy as np
from PIL import Image
from utils.show import draw_fragments_2d

def merge_small_object(seg, threshold=5, window=5):
    uid, uc = np.unique(seg, return_counts=True)
    for (ids, size) in zip(uid, uc):
        if size > threshold:
            continue
        # print(seg.shape)
        # print(ids)
        # print(np.where(seg == ids))
        pos_x, pos_y = np.where(seg == ids)
        pos_x = int(np.sum(pos_x) // np.size(pos_x))
        pos_y = int(np.sum(pos_y) // np.size(pos_y))
        pos_x = pos_x - window // 2
        pos_y = pos_y - window // 2
        seg_crop = seg[pos_x:pos_x+window, pos_y:pos_y+window]
        temp_uid, temp_uc = np.unique(seg_crop, return_counts=True)
        rank = np.argsort(-temp_uc)
        if len(temp_uc) > 2:
            if temp_uid[rank[0]] == 0:
                if temp_uid[rank[1]] == ids:
                    max_ids = temp_uid[rank[2]]
                else:
                    max_ids = temp_uid[rank[1]]
            else:
                max_ids = temp_uid[rank[0]]
            seg[seg==ids] = max_ids
    return seg

def merge_func(seg, step=4):
    seg = merge_small_object(seg)
    seg = merge_small_object(seg, threshold=20, window=11)
    seg = merge_small_object(seg, threshold=50, window=11)
    seg = merge_small_object(seg, threshold=300, window=21)
    return seg

if __name__ == '__main__':
    in_path = '../inference/2021-06-01--09-01-37_cvppp_affs_standard/validation/affs_25500/seg.hdf'
    f = h5py.File(in_path, 'r')
    seg = f['main'][:]
    f.close()

    seg1 = seg[2]
    print(seg1.shape)

    seg_color = draw_fragments_2d(seg1)
    cv2.imwrite('./seg.png', seg_color)
    seg1 = merge_small_object(seg1)
    seg_color = draw_fragments_2d(seg1)
    cv2.imwrite('./seg1.png', seg_color)
    seg1 = merge_small_object(seg1, threshold=20, window=11)
    seg_color = draw_fragments_2d(seg1)
    cv2.imwrite('./seg2.png', seg_color)
    seg1 = merge_small_object(seg1, threshold=50, window=11)
    seg_color = draw_fragments_2d(seg1)
    cv2.imwrite('./seg3.png', seg_color)
    seg1 = merge_small_object(seg1, threshold=300, window=21)
    seg_color = draw_fragments_2d(seg1)
    cv2.imwrite('./seg4.png', seg_color)

    # uid, uc = np.unique(seg1, return_counts=True)
    # print(uid)
    # print(uc)
    # print(sum(uc<20))

    # seg_color = draw_fragments_2d(seg1)
    # cv2.imwrite('./seg.png', seg_color)
