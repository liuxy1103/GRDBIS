import torch
import numpy as np

def simple_augment_torch(data, rule):
    assert np.size(rule) == 3
    assert len(data.shape) == 3
    # x reflection
    if rule[0]:
        data = torch.flip(data, [2])
    # y reflection
    if rule[1]:
        data = torch.flip(data, [1])
    # transpose in xy
    if rule[2]:
        data = data.permute(0, 2, 1)
    return data


def simple_augment_reverse_torch(data, rule):
    assert np.size(rule) == 3
    assert len(data.shape) == 3
    # transpose in xy
    if rule[2]:
        data = data.permute(0, 2, 1)
    # y reflection
    if rule[1]:
        data = torch.flip(data, [1])
    # x reflection
    if rule[0]:
        data = torch.flip(data, [2])
    return data


def convert_consistency_flip(gt, rules):
    B, C, H, W = gt.shape
    gt = gt.detach().clone()
    rules = rules.data.cpu().numpy().astype(np.uint8)
    out_gt = []
    for k in range(B):
        gt_temp = gt[k]
        rule = rules[k]
        gt_temp = simple_augment_reverse_torch(gt_temp, rule)
        out_gt.append(gt_temp)
    out_gt = torch.stack(out_gt, dim=0)
    return out_gt


class Filp_EMA(object):
    def __init__(self):
        super(Filp_EMA, self).__init__()

    def __call__(self, data):
        rule = np.random.randint(2, size=3)
        data = simple_augment_torch(data, rule)
        return data, rule