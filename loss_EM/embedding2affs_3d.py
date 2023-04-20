import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def embedding_loss_norm1(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    embedding = F.normalize(embedding, p=2, dim=1)
    B, C, D, H, W = embedding.shape

    affs0 = torch.sum(embedding[:, :, shift:, :, :]*embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    affs0 = (affs0 + 1) / 2
    affs0 = torch.clamp(affs0, 0.0, 1.0)
    loss0 = criterion(affs0, target[:, 0:1, shift:, :, :], weightmap[:, 0:1, shift:, :, :])
    # loss0 = criterion(affs0, target[:, 0:1, shift:, :, :])

    affs1 = torch.sum(embedding[:, :, :, shift:, :]*embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    affs1 = (affs1 + 1) / 2
    affs1 = torch.clamp(affs1, 0.0, 1.0)
    loss1 = criterion(affs1, target[:, 1:2, :, shift:, :], weightmap[:, 1:2, :, shift:, :])
    # loss1 = criterion(affs1, target[:, 1:2, :, shift:, :])

    affs2 = torch.sum(embedding[:, :, :, :, shift:]*embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
    affs2 = (affs2 + 1) / 2
    affs2 = torch.clamp(affs2, 0.0, 1.0)
    loss2 = criterion(affs2, target[:, 2:3, :, :, shift:], weightmap[:, 2:3, :, :, shift:])
    # loss2 = criterion(affs2, target[:, 2:3, :, :, shift:])

    loss = affs0_weight * loss0 + loss1 + loss2

    affs = torch.zeros_like(target)
    affs[:, 0:1, shift:, :, :] = affs0
    affs[:, 1:2, :, shift:, :] = affs1
    affs[:, 2:3, :, :, shift:] = affs2

    if fill:
        affs[:, 0, :shift, :, :] = affs0[:, 0, :shift, :, :]
        affs[:, 1, :, :shift, :] = affs1[:, 0, :, :shift, :]
        affs[:, 2, :, :, :shift] = affs2[:, 0, :, :, :shift]
    return loss, affs

def embedding_loss_norm1_relu(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    embedding = F.normalize(embedding, p=2, dim=1)
    B, C, D, H, W = embedding.shape

    affs0 = torch.sum(embedding[:, :, shift:, :, :]*embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    affs0 = F.relu(affs0)
    affs0 = torch.clamp(affs0, 0.0, 1.0)
    loss0 = criterion(affs0, target[:, 0:1, shift:, :, :], weightmap[:, 0:1, shift:, :, :])
    # loss0 = criterion(affs0, target[:, 0:1, shift:, :, :])

    affs1 = torch.sum(embedding[:, :, :, shift:, :]*embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    affs1 = F.relu(affs1)
    affs1 = torch.clamp(affs1, 0.0, 1.0)
    loss1 = criterion(affs1, target[:, 1:2, :, shift:, :], weightmap[:, 1:2, :, shift:, :])
    # loss1 = criterion(affs1, target[:, 1:2, :, shift:, :])

    affs2 = torch.sum(embedding[:, :, :, :, shift:]*embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
    affs2 = F.relu(affs2)
    affs2 = torch.clamp(affs2, 0.0, 1.0)
    loss2 = criterion(affs2, target[:, 2:3, :, :, shift:], weightmap[:, 2:3, :, :, shift:])
    # loss2 = criterion(affs2, target[:, 2:3, :, :, shift:])

    loss = affs0_weight * loss0 + loss1 + loss2

    affs = torch.zeros_like(target)
    affs[:, 0:1, shift:, :, :] = affs0
    affs[:, 1:2, :, shift:, :] = affs1
    affs[:, 2:3, :, :, shift:] = affs2

    return loss, affs

def embedding_single_offset(embedding, order, shift=1):
    B, C, D, H, W = embedding.shape
    embedding_shift = torch.zeros_like(embedding)
    if order == 0:
        embedding_shift[:, :, shift:, :, :] = embedding[:, :, :D-shift, :, :]
    elif order == 1:
        embedding_shift[:, :, :, shift:, :] = embedding[:, :, :, :H-shift, :]
    elif order == 2:
        embedding_shift[:, :, :, :, shift:] = embedding[:, :, :, :, :W-shift]
    else:
        raise NotImplementedError
    
    affs_temp = torch.sum(embedding_shift*embedding, dim=1, keepdim=False)
    affs_temp = (affs_temp + 1) / 2
    affs_temp = torch.clamp(affs_temp, 0.0, 1.0)

    if order == 0:
        affs_temp[:, :shift, :, :] = 0.0
    elif order == 1:
        affs_temp[:, :, :shift, :] = 0.0
    elif order == 2:
        affs_temp[:, :, :, :shift] = 0.0
    else:
        raise NotImplementedError
    return affs_temp

def embedding_single_offset_cos(embedding, order, shift=1, dis=nn.CosineSimilarity(dim=1, eps=1e-6)):
    B, C, D, H, W = embedding.shape
    embedding_shift = torch.zeros_like(embedding)
    if order == 0:
        embedding_shift[:, :, shift:, :, :] = embedding[:, :, :D-shift, :, :]
    elif order == 1:
        embedding_shift[:, :, :, shift:, :] = embedding[:, :, :, :H-shift, :]
    elif order == 2:
        embedding_shift[:, :, :, :, shift:] = embedding[:, :, :, :, :W-shift]
    else:
        raise NotImplementedError
    
    affs_temp = dis(embedding_shift, embedding)
    affs_temp = (affs_temp + 1) / 2
    affs_temp = torch.clamp(affs_temp, 0.0, 1.0)

    if order == 0:
        affs_temp[:, :shift, :, :] = 0.0
    elif order == 1:
        affs_temp[:, :, :shift, :] = 0.0
    elif order == 2:
        affs_temp[:, :, :, :shift] = 0.0
    else:
        raise NotImplementedError
    return affs_temp


def embedding_loss_norm2(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    embedding = F.normalize(embedding, p=2, dim=1)
    affs = torch.zeros_like(target)

    for i in range(3):
        affs[:, i] = embedding_single_offset(embedding, i, shift=shift)

    # mask target
    mask = torch.ones_like(target)
    mask[:, 0, :shift, :, :] = 0.0
    mask[:, 1, :, :shift, :] = 0.0
    mask[:, 2, :, :, :shift] = 0.0
    target = target * mask
    weightmap = weightmap * mask

    loss = criterion(affs, target, weightmap)

    # if fill:
    #     affs[:, 1, :, :shift, :] = affs[:, 1, :, shift:shift*2, :]
    #     affs[:, 2, :, :, :shift] = affs[:, 2, :, :, shift:shift*2]
    #     affs[:, 0, :shift, :, :] = affs[:, 0, shift:shift*2, :, :]
    return loss, affs

def embedding_loss_norm3(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    '''
    Based on embedding_loss_norm2, but use nn.CosineSimilarity to compute cosine distance
    '''
    dis = nn.CosineSimilarity(dim=1, eps=1e-6)
    affs = torch.zeros_like(target)

    for i in range(3):
        affs[:, i] = embedding_single_offset_cos(embedding, i, shift=shift, dis=dis)

    # mask target
    mask = torch.ones_like(target)
    mask[:, 0, :shift, :, :] = 0.0
    mask[:, 1, :, :shift, :] = 0.0
    mask[:, 2, :, :, :shift] = 0.0
    target = target * mask
    weightmap = weightmap * mask

    loss = criterion(affs, target, weightmap)

    return loss, affs

def embedding_loss_norm4(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    '''
    Based on embedding_loss_norm2, but use to compute affinity with 12 channels
    '''
    embedding = F.normalize(embedding, p=2, dim=1)
    affs = torch.zeros_like(target)
    shifts = [1, 1, 1, 2, 3, 3, 3, 9, 9, 4, 27, 27]

    for i, shift in enumerate(shifts):
        affs[:, i] = embedding_single_offset(embedding, i%3, shift=shift)

    # mask target
    mask = torch.ones_like(target)
    mask[:, 0, :1, :, :] = 0.0
    mask[:, 1, :, :1, :] = 0.0
    mask[:, 2, :, :, :1] = 0.0
    target = target * mask
    weightmap = weightmap * mask

    loss = criterion(affs, target, weightmap)

    return loss, affs

def embedding_single_offset_loss(embedding, order, shift, target, weightmap, criterion):
    B, C, D, H, W = embedding.shape
    order_shift = order % 3
    if order_shift == 0:
        affs_temp = torch.sum(embedding[:, :, shift:, :, :]*embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    elif order_shift == 1:
        affs_temp = torch.sum(embedding[:, :, :, shift:, :]*embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    elif order_shift == 2:
        affs_temp = torch.sum(embedding[:, :, :, :, shift:]*embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
    else:
        raise NotImplementedError
    
    affs_temp = (affs_temp + 1) / 2
    affs_temp = torch.clamp(affs_temp, 0.0, 1.0)

    if order_shift == 0:
        loss_temp = criterion(affs_temp, target[:, order:order+1, shift:, :, :], weightmap[:, order:order+1, shift:, :, :])
    elif order_shift == 1:
        loss_temp = criterion(affs_temp, target[:, order:order+1, :, shift:, :], weightmap[:, order:order+1, :, shift:, :])
    elif order_shift == 2:
        loss_temp = criterion(affs_temp, target[:, order:order+1, :, :, shift:], weightmap[:, order:order+1, :, :, shift:])
    else:
        raise NotImplementedError

    return loss_temp, affs_temp

def embedding_loss_norm5(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    '''
    Based on embedding_loss_norm4, but compute each channel separately 
    '''
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

def embedding_single_offset_loss_relu(embedding, order, shift, target, weightmap, criterion):
    B, C, D, H, W = embedding.shape
    order_shift = order % 3
    if order_shift == 0:
        affs_temp = torch.sum(embedding[:, :, shift:, :, :]*embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    elif order_shift == 1:
        affs_temp = torch.sum(embedding[:, :, :, shift:, :]*embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    elif order_shift == 2:
        affs_temp = torch.sum(embedding[:, :, :, :, shift:]*embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
    else:
        raise NotImplementedError
    
    # affs_temp = (affs_temp + 1) / 2
    affs_temp = F.relu(affs_temp)
    affs_temp = torch.clamp(affs_temp, 0.0, 1.0)

    if order_shift == 0:
        loss_temp = criterion(affs_temp, target[:, order:order+1, shift:, :, :], weightmap[:, order:order+1, shift:, :, :])
    elif order_shift == 1:
        loss_temp = criterion(affs_temp, target[:, order:order+1, :, shift:, :], weightmap[:, order:order+1, :, shift:, :])
    elif order_shift == 2:
        loss_temp = criterion(affs_temp, target[:, order:order+1, :, :, shift:], weightmap[:, order:order+1, :, :, shift:])
    else:
        raise NotImplementedError

    return loss_temp, affs_temp

def embedding_loss_norm5_relu(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    '''
    Based on embedding_loss_norm4, but compute each channel separately 
    '''
    embedding = F.normalize(embedding, p=2, dim=1)

    affs = torch.zeros_like(target)
    shifts = [1, 1, 1, 2, 3, 3, 3, 9, 9, 4, 27, 27]

    loss = 0
    for i, shift in enumerate(shifts):
        loss_temp, affs_temp = embedding_single_offset_loss_relu(embedding, i, shift, target, weightmap, criterion)
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

if __name__ == '__main__':
    embedding = torch.randn((2, 16, 18, 160, 160))
    target = torch.randn((2, 3, 18, 160, 160))
    target[target<=0.5] = 0
    target[target>0.5] = 1
    weightmap = torch.randn((2, 3, 18, 160, 160))
    criterion = nn.BCELoss()

    loss, affs = embedding_loss_norm2(embedding, target, weightmap, criterion)
    print(loss)
    print(affs.shape)
