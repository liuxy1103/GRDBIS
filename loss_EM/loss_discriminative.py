import torch
import torch.nn as nn
import torch.nn.functional as F


def discriminative_loss(embedding, seg_gt, background=False, delta_v=0.5, delta_d=1.5, alpha=1, beta=1, gama=0.001):
    batch_size = embedding.shape[0]
    embed_dim = embedding.shape[1]
    var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    
    for b in range(batch_size):
        embedding_b = embedding[b] # (embed_dim, D, H, W)
        seg_gt_b = seg_gt[b] # (D, H, W)

        labels = torch.unique(seg_gt_b) # list
        if not background:
            labels = labels[labels!=0] # list
        num_id = len(labels) # int
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
            seg_mask_i = (seg_gt_b == idx) # (D, H, W)
            if not seg_mask_i.any():
                continue
            embedding_i = embedding_b[:, seg_mask_i] # (embed_dim, Nc)
            mean_i = torch.mean(embedding_i, dim=1) # (embed_dim)
            centroid_mean.append(mean_i)

            # ---------- var_loss -------------
            var_loss = var_loss + torch.mean( F.relu(torch.norm(embedding_i-mean_i.reshape(embed_dim,1), dim=0) - delta_v)**2 ) / num_id
        centroid_mean = torch.stack(centroid_mean) # (num_id, embed_dim)

        if num_id > 1:
            centroid_mean1 = centroid_mean.reshape(-1, 1, embed_dim) # (num_id, 1, embed_dim)
            centroid_mean2 = centroid_mean.reshape(1, -1, embed_dim) # (1, num_id, embed_dim)
            dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # (num_id, num_id)
            dist = dist + torch.eye(num_id, dtype=dist.dtype, device=dist.device) * (2*delta_d)  # diagonal elements are 0, now mask above 2*delta_d

            # divided by two for double calculated loss above, for implementation convenience
            dist_loss = dist_loss + torch.sum(F.relu(2*delta_d - dist)**2) / (num_id * (num_id-1)) / 2

        # reg_loss is not used in original paper
        reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

    var_loss = var_loss / batch_size
    dist_loss = dist_loss / batch_size
    reg_loss = reg_loss / batch_size

    Loss  = alpha*var_loss + beta*dist_loss + gama*reg_loss
    return Loss

if __name__ == '__main__':
    embedding = torch.randn((4, 16, 18, 160, 160))
    seg_gt = torch.randint(low=0, high=5, size=(4, 18, 160, 160))
    loss = discriminative_loss(embedding, seg_gt, background=True)
    print(loss)