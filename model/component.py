import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
import math

__all__ = ['MoE']


class MoE(nn.Module):

    def __init__(self, groups=8, in_channels=512):
        super(MoE, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gap_weight = nn.Sequential(
            nn.Conv2d(in_channels, 8 * self.groups, groups=groups, kernel_size=1, padding=0),
            nn.BatchNorm2d(8 * self.groups),
            nn.Tanh(),
            nn.Conv2d(8 * self.groups, self.groups, groups=groups, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.groups)
        )
        self.bn = nn.BatchNorm2d(self.groups)
        self.sigmoid = nn.Sigmoid()
        self.bn2 = nn.BatchNorm2d(self.groups)

    def forward(self, x):
        # the input is feature map after group conv
        b, c, h, w = x.detach().size()
        x = x.contiguous()
        n = c // self.groups

        # weighted global avg pooling
        pool_weights = self.gap_weight(x)  # b,g,h,w
        pool_weights = pool_weights.view(b, self.groups, -1)
        pool_weights = F.softmax(pool_weights, dim=2)
        pool_weights = pool_weights.view(b, self.groups, h, w)
        weighted_x = x.view(b, self.groups, n, h, w)  # b,g,c/g,h,w
        weighted_x = pool_weights.unsqueeze(2) * weighted_x
        weighted_x = weighted_x.view(b, c, h, w)
        x_global_avg_pooling = torch.sum(weighted_x, dim=(2, 3), keepdim=True)  # b,c,1,1
        x_splits = x_global_avg_pooling.view(b, self.groups, n)  # b,g,c/g
        x_splits = x_splits.view(-1, 1, n)  # b*g,1,c/g

        # dot product phase 1
        x_s = x.view(b * self.groups, n, -1)  # b*g,c/g,h*w
        dot_product = torch.bmm(x_splits, x_s)  # b*g,1,h*w
        dot_product = dot_product.view(b, self.groups, h, w)  # b,g,h,w
        dot_product = self.bn(dot_product)
   #     sigmoid_mask = F.softmax(dot_product.view(dot_product.size(0), dot_product.size(1), -1), dim=2)
        sigmoid_mask = self.sigmoid(dot_product)
        sigmoid_mask = sigmoid_mask.repeat(1, n, 1, 1).view(b, n, self.groups, h, w).permute(0, 2, 1, 3, 4)
        sigmoid_mask = sigmoid_mask.contiguous().view(b, c, h, w)  # b,c,h,w
        y = sigmoid_mask * x_global_avg_pooling + x
        
        # routing between heads
        hub_feature = x_splits.view(b, self.groups, n)  # b,g,c/g
        path_matrix = torch.bmm(hub_feature, hub_feature.permute(0, 2, 1))  # b,g,g
        path_matrix = torch.nn.functional.normalize(path_matrix, dim=2)
        path_matrix = F.softmax(path_matrix*math.sqrt(self.groups), dim=2)
       # print(path_matrix)
        x_global_avg_pooling = torch.bmm(path_matrix, hub_feature)
        x_global_avg_pooling = x_global_avg_pooling.view(b, c, 1, 1)
        

        # dot product phase 2
        dot_product2 = torch.bmm(x_global_avg_pooling.view(-1, 1, n), y.view(b * self.groups, n, -1))  # b*g,1,h*w
        dot_product2 = dot_product2.view(b, self.groups, h, w)  # b,g,h,w
        dot_product2 = self.bn2(dot_product2)
    #    sigmoid_mask = F.softmax(dot_product2.view(dot_product2.size(0), dot_product2.size(1), -1), dim=2)
        sigmoid_mask = self.sigmoid(dot_product2)
        sigmoid_mask = sigmoid_mask.repeat(1, n, 1, 1).view(b, n, self.groups, h, w).permute(0, 2, 1, 3, 4)
        sigmoid_mask = sigmoid_mask.contiguous().view(b, c, h, w)  # b,c,h,w
        y = sigmoid_mask * x_global_avg_pooling + y
        
        pool_weights = pool_weights.view(b, self.groups, -1)  # b,g,h*w
        pool_weights, _ = torch.max(pool_weights, dim=1)
        loss = torch.tensor(self.groups) - torch.sum(pool_weights, dim=1)

        return y, loss
