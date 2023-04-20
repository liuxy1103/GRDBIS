import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def segment_sum(data, segment_ids):
    """
    Analogous to tf.segment_sum (https://www.tensorflow.org/api_docs/python/tf/math/segment_sum).

    :param data: A pytorch tensor of the data for segmented summation.
    :param segment_ids: A 1-D tensor containing the indices for the segmentation.
    :return: a tensor of the same type as data containing the results of the segmented summation.
    """
    if not all(segment_ids[i] <= segment_ids[i + 1] for i in range(len(segment_ids) - 1)):
        raise AssertionError("elements of segment_ids must be sorted")

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError("segment_ids should be the same size as dimension 0 of input.")

    num_segments = len(torch.unique(segment_ids))
    return unsorted_segment_sum(data, segment_ids, num_segments)


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long().to(data.device)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape, device=data.device)
    tensor = tensor.scatter_add(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


def embedding_loss_single_example(embedding,
                                  label_map,
                                  neighbor,
                                  include_bg=True):
    label_flat = label_map.reshape([-1])
    embedding_flat = embedding.reshape([-1, embedding.shape[-1]])
    embedding_flat = F.normalize(embedding_flat, p=2, dim=1)
    dis = nn.CosineSimilarity(dim=1, eps=1e-6)

    if not include_bg:
        label_mask = (label_flat > 0)
        label_flat = label_flat[label_mask]
        embedding_flat = embedding_flat[label_mask, :]

    # unique_labels, unique_id = torch.unique(label_flat, sorted=True, return_inverse=True)
    # counts = torch.zeros_like(unique_labels)
    # for i in range(len(unique_labels)):
    #     counts[i] = torch.sum((label_flat == unique_labels[i]).int())
    unique_labels, unique_id, counts = torch.unique(label_flat, sorted=True, return_inverse=True, return_counts=True)
    counts = counts.float().reshape([-1, 1])
    segmented_sum = unsorted_segment_sum(embedding_flat, unique_id, len(unique_labels))
    mu = F.normalize(segmented_sum/counts, p=2, dim=1)
    mu_expand = mu[unique_id]

    loss_inner = torch.mean(dis(mu_expand, embedding_flat))

    instance_num = len(unique_labels)
    mu_interleave = mu.repeat(instance_num, 1)
    mu_rep = mu.repeat(1, instance_num)
    mu_rep = mu_rep.reshape([instance_num*instance_num, -1])

    loss_inter = torch.abs(1 - dis(mu_interleave, mu_rep))

    neighbor = neighbor.long()
    bg = torch.zeros((neighbor.shape[0], 1), dtype=neighbor.dtype, device=neighbor.device)
    neighbor = torch.cat([bg, neighbor], dim=1)
    dep = instance_num if include_bg else instance_num + 1

    adj_indicator = F.one_hot(neighbor, num_classes=dep)
    adj_indicator = torch.sum(adj_indicator, dim=1)
    adj_indicator = (adj_indicator > 0).float()

    temp = torch.zeros((1), dtype=neighbor.dtype, device=neighbor.device)
    bg_indicator = F.one_hot(temp, num_classes=dep).float()
    bg_indicator = 1.0 - bg_indicator
    bg_indicator = bg_indicator.reshape([1, -1])
    indicator = torch.cat([bg_indicator, adj_indicator], dim=0)

    # indicator = indicator[unique_labels, :]
    # indicator = indicator[:, unique_labels]
    indicator = indicator.index_select(0, unique_labels.long())
    indicator = indicator.index_select(1, unique_labels.long())
    inter_mask = indicator.reshape([-1, 1])

    loss_inter = torch.sum(loss_inter * inter_mask) / (torch.sum(inter_mask) + 1e12)

    loss = loss_inner + loss_inter
    return loss


def local_embedding(embedding, seg_gt, neighbor, include_bg=True, norm=True):
    if norm:
        embedding = F.normalize(embedding, p=2, dim=1)
    batch_size = embedding.shape[0]
    embedding = embedding.permute(0,2,3,1)
    loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

    for b in range(batch_size):
        loss_single = embedding_loss_single_example(embedding[b],
                                                    seg_gt[b],
                                                    neighbor[b],
                                                    include_bg)

        loss += loss_single
    
    loss = loss / batch_size
    return loss


if __name__ == '__main__':
    embedding = torch.randn((4, 16, 544, 544))
    seg_gt = torch.randint(low=0, high=16, size=(4, 544, 544))
    neighbor = torch.randint(low=0, high=16, size=(4, 50, 32))
    loss = local_embedding(embedding, seg_gt, neighbor)
    print(loss)