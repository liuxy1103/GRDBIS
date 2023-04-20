import waterz
import mahotas
import numpy as np

def relabel(seg):
    # get the unique labels
    uid = np.unique(seg)
    # ignore all-background samples
    if len(uid)==1 and uid[0] == 0:
        return seg

    uid = uid[uid > 0]
    mid = int(uid.max()) + 1 # get the maximum label for the segment

    # create an array from original segment id to reduced id
    m_type = seg.dtype
    mapping = np.zeros(mid, dtype=m_type)
    mapping[uid] = np.arange(1, len(uid) + 1, dtype=m_type)
    return mapping[seg]

def watershed(affs):
    # affs_xy = 1.0 - 0.5*(affs[0] + affs[1])
    affs = 1.0 - affs
    affs_xy = np.maximum(affs[0], affs[1])

    distance = mahotas.distance(affs_xy<0.5)
    maxima = mahotas.regmax(distance)
    seeds, num_seeds = mahotas.label(maxima)
    fragments = mahotas.cwatershed(affs_xy, seeds)
    return fragments

def get_seeds(boundary, method='grid', next_id=1, radius=5, seed_distance=10):
    if method == 'grid':
        height = boundary.shape[0]
        width  = boundary.shape[1]
        seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x*num_seeds_y
        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y,num_seeds_x))

    if method == 'minima':
        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    if method == 'maxima_distance':
        Bc = np.ones((radius,radius))
        distance = mahotas.distance(boundary<0.5)
        maxima = mahotas.regmax(distance, Bc=Bc)
        seeds, num_seeds = mahotas.label(maxima, Bc=Bc)
        seeds += next_id
        seeds[seeds==next_id] = 0

    return seeds, num_seeds

def gen_fragment(affs, radius=5):
    # boundary = 1.0 - 0.5*(affs[0] + affs[1])
    boundary = 1.0 - np.minimum(affs[0], affs[1])
    seeds, _ = get_seeds(boundary, next_id=1, radius=radius, method='maxima_distance')
    fragments = mahotas.cwatershed(boundary, seeds)
    return fragments

def seg_waterz(affs, mask=None):
    _, h, w = affs.shape
    # fragments = watershed(affs).astype(np.uint64)
    fragments = gen_fragment(affs).astype(np.uint64)
    if mask is not None:
        fragments[mask == 0] = 0
    fragments = fragments[np.newaxis, ...]
    affs_expend = np.zeros((3, 1, h, w), dtype=np.float32)
    affs_expend[1:, 0] = affs
    # sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
    segmentation = list(waterz.agglomerate(affs_expend, [0.50],
                                        fragments=fragments,
                                        scoring_function=sf,
                                        discretize_queue=256))[0]
    # segmentation = relabel(segmentation).astype(np.uint8)
    segmentation = np.squeeze(segmentation)
    fragments = np.squeeze(fragments)
    return segmentation, fragments
