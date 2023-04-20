# -*- coding: utf-8 -*-
# @Time : 2021/3/30
# @Author : Xiaoyu Liu
# @Email : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm

import torch
import random
import numpy as np
from pathlib import Path
import SimpleITK as sitk


def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)
def remove_list(list1, list2):
    out = []
    for k in list1:
        if k in list2:
            continue
        out.append(k)
    return out

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def log_args(args,log):
    args_info = "\n##############\n"
    for key in args.__dict__:
        args_info = args_info+(key+":").ljust(25)+str(args.__dict__[key])+"\n"
    args_info += "##############"
    log.info(args_info)

def save_nii(img,save_name):
    nii_image = sitk.GetImageFromArray(img)
    name = str(save_name).split("/")
    sitk.WriteImage(nii_image,str(save_name))
    print(name[-1]+" saving finished!")


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=40.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(model, ema_model, alpha=0.99, global_step=0):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def _adjust_learning_rate(optimizer, i_iter, learning_rate, max_iters, power):
    lr = lr_poly(learning_rate, i_iter, max_iters, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate(optimizer, i_iter, learning_rate, max_iters, power):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, learning_rate, max_iters, power)

def adjust_learning_rate_discriminator(optimizer, i_iter, learning_rate, max_iters, power):
    _adjust_learning_rate(optimizer, i_iter, learning_rate, max_iters, power)