import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def embedding_single_offset_loss(embedding, offset, target, weightmap, mask, criterion, mode='cos'):
    embedding_shift = torch.roll(embedding, shifts=tuple(offset), dims=(2, 3))
    # affs_temp = dis(embedding_shift, embedding)
    if mode == 'cos':
        affs_temp = torch.sum(embedding_shift * embedding, dim=1)
        affs_temp = (affs_temp + 1) / 2
    else:
        affs_temp = torch.sum((embedding_shift - embedding)**2, dim=1)
        affs_temp = 1 - affs_temp / 4.0
    affs_temp = torch.clamp(affs_temp, 0.0, 1.0)

    loss_temp = criterion(affs_temp*mask, target*mask, weightmap)
    return loss_temp, affs_temp

def embedding_loss(embedding, target, weightmap, mask, criterion, offsets, affs0_weight=1, mode='cos'):
    embedding = F.normalize(embedding, p=2, dim=1)
    # dis = nn.CosineSimilarity(dim=1, eps=1e-6)
    mask = mask.float()

    affs = torch.zeros_like(target)
    loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    for i, offset in enumerate(offsets):
        shift_off = [-x for x in offset]
        loss_temp, affs_temp = embedding_single_offset_loss(embedding, shift_off, target[:,i], weightmap[:,i], mask[:,i], criterion, mode=mode)
        if i < 2:
            loss += loss_temp * affs0_weight
        else:
            loss += loss_temp
        affs[:, i] = affs_temp
    return loss, affs

def embedding_single_offset(embedding, offset, mode='cos'):
    embedding_shift = torch.roll(embedding, shifts=tuple(offset), dims=(2, 3))
    # affs_temp = dis(embedding_shift, embedding)
    if mode == 'cos':
        affs_temp = torch.sum(embedding_shift * embedding, dim=1)
        affs_temp = (affs_temp + 1) / 2
    else:
        affs_temp = torch.sum((embedding_shift - embedding)**2, dim=1)
        affs_temp = 1 - affs_temp / 4.0
    affs_temp = torch.clamp(affs_temp, 0.0, 1.0)
    return affs_temp

def embedding2affs(embedding, offsets, mode='cos'):
    embedding = F.normalize(embedding, p=2, dim=1)
    B, C, H, W = embedding.shape
    # dis = nn.CosineSimilarity(dim=1, eps=1e-6)
    affs = torch.zeros((B, len(offsets), H, W), dtype=embedding.dtype, device=embedding.device)
    for i, offset in enumerate(offsets):
        shift_off = [-x for x in offset]
        affs[:, i] = embedding_single_offset(embedding, shift_off, mode=mode)
    return affs

def ema_embedding_single_offset_loss(embedding, ema_embedding, offset, target, weightmap, mask, criterion, mode='cos'):
    ema_embedding = torch.roll(ema_embedding, shifts=tuple(offset), dims=(2, 3))
    # affs_temp = dis(embedding_shift, embedding)
    if mode == 'cos':
        affs_temp = torch.sum(ema_embedding * embedding, dim=1)
        affs_temp = (affs_temp + 1) / 2
    else:
        affs_temp = torch.sum((ema_embedding - embedding)**2, dim=1)
        affs_temp = 1 - affs_temp / 4.0
    affs_temp = torch.clamp(affs_temp, 0.0, 1.0)

    loss_temp = criterion(affs_temp*mask, target*mask, weightmap)
    return loss_temp, affs_temp

def ema_embedding_loss(embedding, ema_embedding, target, weightmap, mask, criterion, offsets, affs0_weight=1, mode='cos'):
    embedding = F.normalize(embedding, p=2, dim=1)
    ema_embedding = F.normalize(ema_embedding, p=2, dim=1)
    # dis = nn.CosineSimilarity(dim=1, eps=1e-6)
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
