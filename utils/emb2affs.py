import numpy as np
import torch
from torch import nn as nn


def shift_tensor(tensor, offset):
    """ Shift a tensor by the given (spatial) offset.
    Arguments:
        tensor [torch.Tensor] - 4D (=2 spatial dims) or 5D (=3 spatial dims) tensor.
            Needs to be of float type.
        offset (tuple) - 2d or 3d spatial offset used for shifting the tensor
    """

    ndim = len(offset)
    assert ndim in (2, 3)
    diff = tensor.dim() - ndim

    # don't pad for the first dimensions
    # (usually batch and/or channel dimension)
    slice_ = diff * [slice(None)]

    # torch padding behaviour is a bit weird.
    # we use nn.ReplicationPadND
    # (torch.nn.functional.pad is even weirder and ReflectionPad is not supported in 3d)
    # still, padding needs to be given in the inverse spatial order

    # add padding in inverse spatial order
    padding = []
    for off in offset[::-1]:
        # if we have a negative offset, we need to shift "to the left",
        # which means padding at the right border
        # if we have a positive offset, we need to shift "to the right",
        # which means padding to the left border
        padding.extend([max(0, off), max(0, -off)])

    # add slicing in the normal spatial order
    for off in offset:
        if off == 0:
            slice_.append(slice(None))
        elif off > 0:
            slice_.append(slice(None, -off))
        else:
            slice_.append(slice(-off, None))

    # pad the spatial part of the tensor with replication padding
    slice_ = tuple(slice_)
    padding = tuple(padding)
    padder = nn.ReplicationPad2d if ndim == 2 else nn.ReplicationPad3d
    padder = padder(padding)
    shifted = padder(tensor)

    # slice the oadded tensor to get the spatially shifted tensor
    shifted = shifted[slice_]
    assert shifted.shape == tensor.shape

    return shifted


def invert_offsets(offsets):
    return [[-off for off in offset] for offset in offsets]


def embedding_distance(embedding1,embedding2): #B,C,H,W  L2 distance
    dis_emb = torch.norm(embedding1 - embedding2, dim=1)
    loss = torch.mean(dis_emb)
    return loss

def embedding_distance_kl(embedding1,embedding2): #B,C,H,W  kl-divergence distance
    loss = nn.functional.kl_div(embedding1,embedding2)
    # dis_emb = torch.norm(embedding1 - embedding2, dim=1)
    # loss = torch.mean(dis_emb)
    return loss

def embeddings_to_affinities(embeddings, offsets, delta=1.5):
    """ Transform embeddings to affinities.
    """
    # shift the embeddings by the offsets and stack them along a new axis
    # we need to shift in the opposite direction of the offsets, so we invert them
    # before applying the shift
    offsets_ = invert_offsets(offsets)
    shifted = torch.cat([shift_tensor(embeddings, off).unsqueeze(1) for off in offsets_], dim=1)
    # substract the embeddings from the shifted embeddings, take the norm and
    # transform to affinities based on the delta distance
    affs = (2 * delta - torch.norm(embeddings.unsqueeze(1) - shifted, dim=2)) / (2 * delta)
    affs = torch.clamp(affs, min=0) ** 2
    return affs



def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8
def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp #(B,C,H,W) normalize
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1) #B,C,N
    return torch.einsum('icm,icn->imn', [feat, feat])

def similarity_delta(feat,delta=1.5):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp #(B,C,H,W) normalize
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1).unsqueeze(-1) #B,C,N,1
    feat_t = feat.transpose(2,3)#B,C,N,1
    affs = (2 * delta - torch.norm(feat - feat_t, dim=2)) / (2 * delta)
    return affs


def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity_delta(f_T) - similarity_delta(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

def CriterionPairWiseforWholeFeatAfterPool(feat_S, feat_T,scale=1):
    total_w, total_h = feat_T.shape[2], feat_T.shape[3]   #(B,C,H,W)
    patch_w, patch_h = int(total_w*scale), int(total_h*scale)
    maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
    # print('affinity matrix shape:',maxpool(feat_S).shape)
    loss = sim_dis_compute(maxpool(feat_S), maxpool(feat_T))
    # loss = embedding_distance(maxpool(feat_S), maxpool(feat_T))
    return loss

def segmentation_to_affinities(segmentation, offsets):
    """ Transform segmentation to affinities.
    Arguments:
        segmentation [torch.tensor] - 4D (2 spatial dims) or 5D (3 spatial dims) segmentation tensor.
            The channel axis (= dimension 1) needs to be a singleton.
        offsets [list[tuple]] - list of offsets for which to compute the affinities.
    """
    assert segmentation.shape[1] == 1
    # shift the segmentation and substract the shifted tensor from the segmentation
    # we need to shift in the opposite direction of the offsets, so we invert them
    # before applying the shift
    offsets_ = invert_offsets(offsets)
    shifted = torch.cat([shift_tensor(segmentation.float(), off) for off in offsets_], dim=1)
    affs = (segmentation - shifted)
    # the affinities are 1, where we had the same segment id (the difference is 0)
    # and 0 otherwise
    affs.eq_(0.)
    return affs

