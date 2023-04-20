import numpy as np
from affogato.affinities import compute_affinities


def gen_affs_official(labels, offsets=[[-1,0],[0,-1]], ignore=False, ignore_id=0, return_mask=False):
    affs, mask = compute_affinities(labels.astype(np.uint64), offsets, ignore, ignore_id)
    if return_mask:
        return affs, mask
    else:
        return affs