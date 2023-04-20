import numpy as np
from scipy.ndimage import shift

def gen_offsets(shift, neighbor=4):
    assert neighbor == 4 or neighbor == 8, 'neigbor must be 4 or 8!'
    if neighbor == 4:
        return [[-shift, 0], [0, -shift]]
    else:
        return [[-shift, 0], [0, -shift], [-shift, -shift], [-shift, shift]]

def multi_offset(shifts, neighbor=4):
    out = []
    for shift in shifts:
        out += gen_offsets(shift, neighbor=neighbor)
    return out

def gen_affs_ours(labels, offsets=[[-1,0],[0,-1]], ignore=False, padding=False):
    n_channels = len(offsets)
    affinities = np.zeros((n_channels,) + labels.shape, dtype=np.float32)
    masks = np.zeros((n_channels,) + labels.shape, dtype=np.uint8)
    for cid, off in enumerate(offsets):
        shift_off = [-x for x in off]
        shifted = shift(labels, shift_off, order=0, prefilter=False)
        mask = np.ones_like(labels)
        mask = shift(mask, shift_off, order=0, prefilter=False)
        dif = labels - shifted
        out = dif.copy()
        out[dif == 0] = 1
        out[dif != 0] = 0
        if ignore:
            out[labels == 0] = 0
            out[shifted == 0] = 0
        if padding:
            out[mask==0] = 1
        else:
            out[mask==0] = 0
        affinities[cid] = out
        masks[cid] = mask
    return affinities, masks
