import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def loss_aff(pred, pred_T,weightmap,affs_mask, criterion, offsets):

    loss = torch.tensor(0, dtype=pred.dtype, device=pred.device)
    for i, offset in enumerate(offsets):
        pred_tmp, pred_T_tmp, mask_tmp, weightmap_tmp = pred[:,i], pred_T[:,i], affs_mask[:,i].float(),weightmap[:,i]

        loss_temp = criterion(pred_tmp * mask_tmp, pred_T_tmp * mask_tmp, weightmap_tmp)
        loss += loss_temp

    return loss
