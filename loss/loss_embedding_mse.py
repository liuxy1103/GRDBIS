import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def embedding_single_offset_loss(embedding, offset, target, weightmap, mask, criterion, mode='ours'):
    embedding_shift = torch.roll(embedding, shifts=tuple(offset), dims=(2, 3))
    if mode == 'ours':
        affs_temp = torch.sum(embedding_shift * embedding, dim=1)
    else:
        dis = nn.CosineSimilarity(dim=1, eps=1e-6)
        affs_temp = dis(embedding_shift, embedding)

    loss_temp = criterion(affs_temp*mask, target*mask, weightmap)
    return loss_temp, affs_temp

def embedding_loss(embedding, target, weightmap, mask, criterion, offsets, affs0_weight=1, mode='ours'):
    if mode == 'ours':
        embedding = F.normalize(embedding, p=2, dim=1)
    mask = mask.float()

    affs = torch.zeros_like(target)
    loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    all_loss = []
    if affs0_weight == 1:
        affs0_weight_factor = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif affs0_weight == 2:
        affs0_weight_factor = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif affs0_weight == 3:
        affs0_weight_factor = [0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        affs0_weight_factor = [affs0_weight, affs0_weight, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    for i, offset in enumerate(offsets):
        shift_off = [-x for x in offset]
        loss_temp, affs_temp = embedding_single_offset_loss(embedding, shift_off, target[:,i], weightmap[:,i], mask[:,i], criterion, mode=mode)
        # loss += loss_temp * affs0_weight_factor[i]
        loss += loss_temp
        # all_loss.append((loss_temp * affs0_weight_factor[i]).item())
        all_loss.append(loss_temp.item())
        # if i < 2:
        #     loss += loss_temp * affs0_weight
        # else:
        #     loss += loss_temp
        affs[:, i] = affs_temp
    return loss, affs, all_loss

def embedding_single_offset(embedding, offset, mode='ours'):
    embedding_shift = torch.roll(embedding, shifts=tuple(offset), dims=(2, 3))
    if mode == 'ours':
        affs_temp = torch.sum(embedding_shift * embedding, dim=1)
    else:
        dis = nn.CosineSimilarity(dim=1, eps=1e-6)
        affs_temp = dis(embedding_shift, embedding)
    return affs_temp

def embedding2affs(embedding, offsets, mode='ours'):
    if mode == 'ours':
        embedding = F.normalize(embedding, p=2, dim=1)
    B, C, H, W = embedding.shape
    affs = torch.zeros((B, len(offsets), H, W), dtype=embedding.dtype, device=embedding.device)
    for i, offset in enumerate(offsets):
        shift_off = [-x for x in offset]
        affs[:, i] = embedding_single_offset(embedding, shift_off, mode=mode)
    return affs

def ema_embedding_single_offset_loss(embedding, ema_embedding, offset, target, weightmap, mask, criterion, mode='ours'):
    ema_embedding = torch.roll(ema_embedding, shifts=tuple(offset), dims=(2, 3))
    if mode == 'ours':
        affs_temp = torch.sum(ema_embedding * embedding, dim=1)
    else:
        dis = nn.CosineSimilarity(dim=1, eps=1e-6)
        affs_temp = dis(ema_embedding, embedding)

    loss_temp = criterion(affs_temp*mask, target*mask, weightmap)
    return loss_temp, affs_temp

def ema_embedding_loss(embedding, ema_embedding, target, weightmap, mask, criterion, offsets, affs0_weight=1, mode='ours'):
    if mode == 'ours':
        embedding = F.normalize(embedding, p=2, dim=1)
        ema_embedding = F.normalize(ema_embedding, p=2, dim=1)
    mask = mask.float()

    affs = torch.zeros_like(target)
    loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    for i, offset in enumerate(offsets):
        shift_off = [-x for x in offset]
        loss_temp, affs_temp = ema_embedding_single_offset_loss(embedding, ema_embedding, shift_off, target[:,i], weightmap[:,i], mask[:,i], criterion, mode=mode)
        if i < 2:
            loss += loss_temp * affs0_weight
        else:
            loss += loss_temp
        affs[:, i] = affs_temp
    return loss, affs