import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def embedding_loss_discriminative_norm1(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    embedding = F.normalize(embedding, p=2, dim=1)
    B, C, D, H, W = embedding.shape
    limited = 1e-6

    affs0 = torch.sum((embedding[:, :, shift:, :, :] - embedding[:, :, :D-shift, :, :])**2, dim=1, keepdim=True)
    sumf0 = torch.sum(target[:, 0:1, shift:, :, :])
    if sumf0 < limited:
        sumf0 = limited
    sumb0 = torch.sum(1 - target[:, 0:1, shift:, :, :])
    if sumb0 < limited:
        sumb0 = limited
    l1 = torch.sum(affs0 * target[:, 0:1, shift:, :, :]) / sumf0
    l2 = torch.sum((1 - affs0 / 4.0) * (1 - target[:, 0:1, shift:, :, :])) / sumb0
    loss0 = l1 + l2
    # print('l1=%.6f, l2=%.6f, l0ss0=%.6f' % (l1.item(), l2.item(), loss0.item()))
    # affs0 = 1 - affs0 / 4.0
    # loss0 = criterion(affs0, target[:, 0:1, shift:, :, :], weightmap[:, 0:1, shift:, :, :])

    affs1 = torch.sum((embedding[:, :, :, shift:, :] - embedding[:, :, :, :H-shift, :])**2, dim=1, keepdim=True)
    sumf1 = torch.sum(target[:, 1:2, :, shift:, :])
    if sumf1 < limited:
        sumf1 = limited
    sumb1 = torch.sum(1 - target[:, 1:2, :, shift:, :])
    if sumb1 < limited:
        sumb1 = limited
    l3 = 10 * torch.sum(affs1 * target[:, 1:2, :, shift:, :]) / sumf1
    l4 = torch.sum((1 - affs1 / 4.0) * (1 - target[:, 1:2, :, shift:, :])) / sumb1
    loss1 = l3 + l4
    # print('l3=%.6f, l4=%.6f, loss1=%.6f' % (l3.item(), l4.item(), loss1.item()))
    # affs1 = 1 - affs1 / 4.0
    # loss1 = criterion(affs1, target[:, 1:2, :, shift:, :], weightmap[:, 1:2, :, shift:, :])

    affs2 = torch.sum((embedding[:, :, :, :, shift:] - embedding[:, :, :, :, :W-shift])**2, dim=1, keepdim=True)
    sumf2 = torch.sum(target[:, 2:3, :, :, shift:])
    if sumf2 < limited:
        sumf2 = limited
    sumb2 = torch.sum(1 - target[:, 2:3, :, :, shift:])
    if sumb2 < limited:
        sumb2 = limited
    l5 = 10 * torch.sum(affs2 * target[:, 2:3, :, :, shift:]) / sumf2
    l6 = torch.sum((1 - affs2 / 4.0) * (1 - target[:, 2:3, :, :, shift:])) / sumb2
    loss2 = l5 + l6
    # print('l5=%.6f, l6=%.6f, loss2=%.6f' % (l5.item(), l6.item(), loss2.item()))
    # affs2 = 1 - affs2 / 4.0
    # loss2 = criterion(affs2, target[:, 2:3, :, :, shift:], weightmap[:, 2:3, :, :, shift:])

    loss = affs0_weight * loss0 + loss1 + loss2

    affs = torch.zeros_like(target)
    affs[:, 0:1, shift:, :, :] = 1 - affs0 / 4.0
    affs[:, 1:2, :, shift:, :] = 1 - affs1 / 4.0
    affs[:, 2:3, :, :, shift:] = 1 - affs2 / 4.0

    return loss, affs


def embedding_single_offset_loss(embedding, order, shift, target, weightmap, criterion):
    B, C, D, H, W = embedding.shape
    order_shift = order % 3
    if order_shift == 0:
        # affs_temp = torch.sum(embedding[:, :, shift:, :, :]*embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
        affs_temp = torch.sum((embedding[:, :, shift:, :, :] - embedding[:, :, :D-shift, :, :])**2,  dim=1, keepdim=True)
        target_temp = target[:, 0:1, shift:, :, :]
    elif order_shift == 1:
        # affs_temp = torch.sum(embedding[:, :, :, shift:, :]*embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
        affs_temp = torch.sum((embedding[:, :, :, shift:, :] - embedding[:, :, :, :H-shift, :])**2,  dim=1, keepdim=True)
        target_temp = target[:, 1:2, :, shift:, :]
    elif order_shift == 2:
        # affs_temp = torch.sum(embedding[:, :, :, :, shift:]*embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
        affs_temp = torch.sum((embedding[:, :, :, :, shift:] - embedding[:, :, :, :, :W-shift])**2,  dim=1, keepdim=True)
        target_temp = target[:, 2:3, :, :, shift:]
    else:
        raise NotImplementedError
    
    loss_temp = torch.sum(affs_temp * target_temp) / torch.sum(target_temp) + \
                torch.sum((1 - affs_temp / 4.0) * (1 - target_temp)) / torch.sum(1 - target_temp)

    # affs_temp = 1 - affs_temp / 4.0
    # affs_temp = torch.clamp(affs_temp, 0.0, 1.0)

    return loss_temp, 1 - affs_temp / 4.0

def embedding_loss_discriminative_norm5(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    embedding = F.normalize(embedding, p=2, dim=1)

    affs = torch.zeros_like(target)
    shifts = [1, 1, 1, 2, 3, 3, 3, 9, 9, 4, 27, 27]

    loss = 0
    for i, shift in enumerate(shifts):
        loss_temp, affs_temp = embedding_single_offset_loss(embedding, i, shift, target, weightmap, criterion)
        if i < 3:
            loss += loss_temp * affs0_weight
        else:
            loss += loss_temp
        if i % 3 == 0:
            affs[:, i:i+1, shift:, :, :] = affs_temp
        elif i % 3 == 1:
            affs[:, i:i+1, :, shift:, :] = affs_temp
        elif i % 3 == 2:
            affs[:, i:i+1, :, :, shift:] = affs_temp
        else:
            raise NotImplementedError

    return loss, affs