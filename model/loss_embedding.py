import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def discriminative_loss(embedding, seg_gt,delta_v=0.5,delta_d=3.0,alpha=1,beta=1,gama=0.001):
    batch_size = embedding.shape[0]
    embed_dim = embedding.shape[1]
    var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    
    for b in range(batch_size):
        embedding_b = embedding[b] # (embed_dim, H, W)
        seg_gt_b = seg_gt[b]

        labels = torch.unique(seg_gt_b)
        labels = labels[labels!=0]
        num_id = len(labels)
        if num_id==0:
            # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
            _nonsense = embedding.sum()
            _zero = torch.zeros_like(_nonsense)
            var_loss = var_loss + _nonsense * _zero
            dist_loss = dist_loss + _nonsense * _zero
            reg_loss = reg_loss + _nonsense * _zero
            continue

        centroid_mean = []
        for idx in labels:
            seg_mask_i = (seg_gt_b == idx)
            if not seg_mask_i.any():
                continue
            embedding_i = embedding_b[:, seg_mask_i] #get positive positions
            # print(embedding_i.shape)
            mean_i = torch.mean(embedding_i, dim=1)  #????
            # print(mean_i.shape)
            centroid_mean.append(mean_i)

            # ---------- var_loss -------------

            var_loss = var_loss + torch.mean( F.relu(torch.norm(embedding_i-mean_i.reshape(embed_dim,1), dim=0) - delta_v)**2 ) / num_id
        centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

        if num_id > 1:
            centroid_mean1 = centroid_mean.reshape(-1, 1, embed_dim)
            centroid_mean2 = centroid_mean.reshape(1, -1, embed_dim)
            dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape (num_id, num_id)
            dist = dist + torch.eye(num_id, dtype=dist.dtype, device=dist.device) * delta_d  # diagonal elements are 0, now mask above delta_d

            # divided by two for double calculated loss above, for implementation convenience
            dist_loss = dist_loss + torch.sum(F.relu(-dist + delta_d)**2) / (num_id * (num_id-1)) / 2

        # reg_loss is not used in original paper
        reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

    var_loss = var_loss / batch_size
    dist_loss = dist_loss / batch_size
    reg_loss = reg_loss / batch_size

    Loss  = alpha*var_loss + beta*dist_loss + gama*reg_loss
    return Loss