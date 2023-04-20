# -*- coding: utf-8 -*-
# @Time : 2022/12/17
# @Author : Xiaoyu Liu
# @Email : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm
#Function: construct a memory bank for feature maps

from pathlib import Path
import SimpleITK as sitk
from skimage.segmentation import slic, mark_boundaries
from skimage import io
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


class cross_image_memory(nn.Module):

    def __init__(self, memory_size, contrast_size, t_channels, img_size):
        super(cross_image_memory, self).__init__()
        # self.s_channels = s_channels
        self.t_channels = t_channels
        self.dim = t_channels
        # self.project_head = nn.Sequential(
        #     nn.Conv2d(s_channels, t_channels, 1, bias=False),
        #     nn.SyncBatchNorm(t_channels),
        #     nn.ReLU(True),
        #     nn.Conv2d(t_channels, t_channels, 1, bias=False)
        # ) # if s_channels!=t_channels
        self.memory_size = memory_size
        # self.update_freq = 16
        self.contrast_size = contrast_size
        self.feature_size_h, self.feature_size_w = img_size,img_size
        self.register_buffer("teacher_feature_queue", torch.randn(self.memory_size, self.dim, self.feature_size_h, self.feature_size_w)) # MxCxHxW
        self.register_buffer("teacher_mask_queue", torch.randn(self.memory_size,1, self.feature_size_h, self.feature_size_w)) # Mx1xHxW
        self.register_buffer("queue_number", torch.zeros(1, dtype=torch.long))  # recoder the number of enqueued samples
        self.iters = 0
    def _dequeue_and_enqueue(self, keys, labels): # keys:#BxCxHxW  labels: #Bx1xHxW
        # teacher_feature_queue = self.teacher_feature_queue  #feature maps
        # teacher_mask_queue = self.teacher_mask_queue # segmentation masks
        
        batch_size, feat_dim, H, W = keys.size()
        
        for bs in range(batch_size):
            this_feat = keys[bs]  #CxHxW
            this_label = labels[bs]#1xHxW
            if self.queue_number < self.memory_size:
                self.teacher_feature_queue[self.queue_number, :] = this_feat
                self.teacher_mask_queue[self.queue_number, :] = this_label
                self.queue_number += torch.tensor(1).cuda()
            else:
                self.teacher_feature_queue[-1, :] = this_feat
                self.teacher_mask_queue[-1, :] = this_label
                self.queue_number = torch.tensor(0).cuda()


    def _sample_negative(self, index):
        cache_size, feat_size, H, W = self.teacher_feature_queue.shape  # MxCxHxW
        contrast_size = index.size(0)
        if contrast_size > self.queue_number:
            contrast_size = self.queue_number
            index = index[:contrast_size]
        X_ = torch.zeros((contrast_size, feat_size, H, W)).float().cuda()
        y_ = torch.zeros((contrast_size, 1, H, W)).float().cuda()
        this_feature = self.teacher_feature_queue[index, :]
        this_mask = self.teacher_mask_queue[index, :]
        return this_feature, this_mask


    def forward(self, t_feats, labels):  #feats: BxCxHxW, labels: Bx1xHxW
        # if self.s_channels != self.t_channels:
        #     s_feats = self.project_head(s_feats)  # project stu emb's channels to match the thecher's
        self.iters = self.iters+1
        labels = labels.float().clone()  # labels: Bx1xHxW
        self._dequeue_and_enqueue(t_feats.detach().clone(), labels.detach().clone()) #BxCxHxW enqueue
        queue_size, _, _, _ = self.teacher_feature_queue.shape
        # print('self.iters:',self.iters)
        if self.iters < self.contrast_size:
            index = torch.arange(self.contrast_size)
        elif self.iters < queue_size:
            perm = torch.randperm(self.queue_number.item())
            index = perm[:self.contrast_size]
        else:
            perm = torch.randperm(queue_size)
            index = perm[:self.contrast_size]
        this_feature, this_mask = self._sample_negative(index)

        return this_feature, this_mask


def pair_wise_sim_map(fea_0, fea_1):
    C, H, W = fea_0.size()

    fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
    fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
    
    sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
    return sim_map_0_1


def calculate_CI_affinity_loss(this_embedding, embedding, embedding_T,loss_type='KL'):
    M,C,H,W = this_embedding.shape
    B,C,H,W = embedding.shape
    this_embedding = F.normalize(this_embedding, p=2, dim=1)
    embedding = F.normalize(embedding, p=2, dim=1)
    embedding_T = F.normalize(embedding_T, p=2, dim=1)
    sim_dis = torch.tensor(0.).cuda()
    for i in range(M):
        for j in range(B):
            avg_pool = nn.AvgPool2d(kernel_size=(7, 7), stride=(7, 7), padding=0, ceil_mode=True)
            feat_this = avg_pool(this_embedding[i])
            feat_S = avg_pool(embedding[j])
            feat_T= avg_pool(embedding_T[j])
            s_sim_map = pair_wise_sim_map(feat_this, feat_S)
            t_sim_map = pair_wise_sim_map(feat_this, feat_T)
            if loss_type=='KL':
                temperature = 1.0
                p_s = F.log_softmax(s_sim_map / temperature, dim=1)
                p_t = F.softmax(t_sim_map / temperature, dim=1)
                sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
            else:
                criterion = nn.MSELoss()
                s1 = torch.prod(torch.tensor(s_sim_map.size()[-1]).float())
                norm_term = (s1).cuda()
                sim_dis_ = torch.sum((s_sim_map - t_sim_map) ** 2)/norm_term
            
            sim_dis += sim_dis_

    return sim_dis


def calclulate_CI_graph_loss(h_list_this, h_list_T, h_list,if_node=False,if_edge_discrepancy=False,if_edge_relation=False,if_neighbor=True,node_weight=1,edge_weight=1,loss_type='KL'):
    "List: h_list_T: #[[NxC,Nxc....layers],[NxC,Nxc....layers],[NxC,Hxc....layers]...batches]"
    "List: edge_list: #[[edge_list,edge_list....layers],[edge_list,edge_list....layers],[edge_list,edge_list....layers]...batches]"
    loss_all = torch.tensor(0.).cuda()
    loss_node_all = torch.tensor(0.).cuda()
    loss_edge_all = torch.tensor(0.).cuda()
    for h_list_T_b,h_list_b in zip(h_list_T,h_list):
        for x_T,x_S in zip(h_list_T_b,h_list_b):
            for h_list_this_b in h_list_this:
                for x_this in h_list_this_b:
                    x_T = torch.cat((x_T,x_this),dim=0)
                    x_S = torch.cat((x_S,x_this),dim=0)
                    "tensor: X: NxF"
                    "List: edge: [(1,2),(2,3)....]]"
                    N = x_T.shape[0]
                    adj = torch.zeros((N,N)).cuda()
                    if if_node:
                        if loss_type=='KL':
                            x_T = F.normalize(x_T, p=2, dim=1)
                            x_S = F.normalize(x_S, p=2, dim=1)
                            temperature = 1.0
                            p_s = F.log_softmax(x_T / temperature, dim=1)
                            p_t = F.softmax(x_S / temperature, dim=1)
                            loss_node = F.kl_div(p_s, p_t, reduction='batchmean')* node_weight
                        else:
                            dis_T_S = torch.norm(x_T-x_S, dim=1)
                            loss_node = torch.mean(dis_T_S) * node_weight
                        loss_all += loss_node
                        loss_node_all += loss_node
                    if if_edge_relation:
                        if loss_type=='KL':
                            x_i = x_T.unsqueeze(1)  # Nx1xf
                            x_j = torch.transpose(x_i, 0, 1)  # 1xNxf
                            x_ij = x_i - x_j  # NxNxf
                            x_T_ij = x_ij.float()
                            ori_shape = x_T_ij.shape
                            x_T_ij = torch.reshape(x_T_ij, (-1, ori_shape[-1]))  # (NxN)xf

                            x_i = x_S.unsqueeze(1)  # Nx1xf
                            x_j = torch.transpose(x_i, 0, 1)  # 1xNxf
                            x_S_ij = x_i - x_j  # NxNxf
                            x_S_ij = x_S_ij.float()
                            ori_shape = x_S_ij.shape
                            x_S_ij = torch.reshape(x_S_ij, (-1, ori_shape[-1]))  # (NxN)xf
                            x_T_ij = F.normalize(x_T_ij, p=2, dim=1)
                            x_S_ij = F.normalize(x_S_ij, p=2, dim=1)
                            temperature = 1.0
                            p_s = F.log_softmax(x_T_ij / temperature, dim=1)
                            p_t = F.softmax(x_S_ij / temperature, dim=1)
                            loss_edge = F.kl_div(p_s, p_t, reduction='batchmean')*edge_weight
                        else:
                            #L2 norm
                            edge_T = calculate_node_similarity(x_T,adj,if_neighbor)# NxN
                            edge_S = calculate_node_similarity(x_S,adj,if_neighbor)# NxN
                            if if_neighbor:
                                loss_edge = torch.sum(torch.abs(edge_T-edge_S))/torch.sum(adj == 1)*edge_weight
                            else:
                                loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / (x_T.shape[0])/ (x_T.shape[0])*edge_weight
                        # loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / torch.sum(adj == 1)
                        loss_all += loss_edge
                        loss_edge_all += loss_edge

    return loss_all/len(h_list_T), loss_node_all/len(h_list_T), loss_edge_all/len(h_list_T)







def region_contrast(x, gt, id):
    """
    calculate region contrast value
    :param x: feature
    :param gt: mask
    :return: value
    """
    smooth = 1.0
    gt = gt[:, 0].unsqueeze(1)
    mask1 = gt[:, 1].unsqueeze(1)

    region0 = torch.sum(x * mask0, dim=(2, 3)) / torch.sum(mask0, dim=(2, 3))
    region1 = torch.sum(x * mask1, dim=(2, 3)) / (torch.sum(mask1, dim=(2, 3)) + smooth)
    return F.cosine_similarity(region0, region1, dim=1)

def adjacent_edge(segments):
    '''
    segments: hxw
    '''

    segments = segments.cpu().squeeze().numpy()
    edge = []
    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])

    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    # Adjacency loops
    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i]<bneighbors[1, i] and bneighbors[0, i]>0:
            edge.append((bneighbors[0, i].item(), bneighbors[1, i].item()))

    return edge



def construct_graph(target_ins,emb_list,if_adjacent=False):
    "target_ins: Bx1xHxW"
    "embedding: list[BxCxHxW]"
    h_list = [] #[[HxC,Hxc....layers],[HxC,Hxc....layers],[HxC,Hxc....layers]...batches]
    edge_list = [] #[[edge_list,edge_list....layers],[edge_list,edge_list....layers],[edge_list,edge_list....layers]...batches]

    for batch in range(target_ins.shape[0]):
        h_list_b = []
        edge_list_b = []
        for emb in emb_list:
            target_ins_b = target_ins[batch]
            emb_b = emb[batch]
            if target_ins_b.shape[-2:] != emb_b.shape[-2:]: #C,H,W
                target_ins_b = F.interpolate(target_ins_b.unsqueeze(0).float(), emb_b.unsqueeze(0).shape[-2:], mode='nearest').squeeze(0)
            if if_adjacent:
                edge = adjacent_edge(target_ins_b) #including background
                edge_list_b.append(edge)
            else:
                edge_list_b.append([(0,0)])
            ins_list = list(torch.unique(target_ins_b))
            ins_list.remove(0)
            ins_list.sort()
            h = torch.cat([torch.mean(emb_b[:,torch.tensor(target_ins_b[0])==id],dim=1).unsqueeze(0) for id in ins_list])
            h_list_b.append(h)
        h_list.append(h_list_b)
        edge_list.append(edge_list_b)

    return h_list,edge_list

def calculate_self_node_similarity(X,Adj,max_id_list_list,if_remove_back=True,delta=1.5):
    "List: X: NxF"
    "List: Adj: NxN"
    loss_all = 0

    for x,adj,max_id_list in zip(X,Adj,max_id_list_list):
        if if_remove_back:
            adj[0, :] = 0
            adj[:, 0] = 0
            for origin in max_id_list[:-1]:
                adj[origin, :] = 0
                adj[:, origin] = 0

        #L2 norm
        x_i = x.unsqueeze(1)  # Nx1xf
        x_j = torch.transpose(x_i, 0, 1)  # 1xNxf
        x_ij = x_i - x_j  # NxNxf
        x_ij = x_ij.float()

        ori_shape = x_ij.shape
        x_ij = torch.reshape(x_ij,(-1,ori_shape[-1])) #(NxN)xf
        x_ij = torch.norm(x_ij,dim=1)

        x_ij = x_ij.reshape(ori_shape[:-1]) # NxN
        x_ij = x_ij * adj.float()  # NxN   #x_ij.max()=1.2
        x_ij = F.relu((2*delta - x_ij)* adj.float()) ** 2 #x_ij.max()=9
        loss = torch.sum(x_ij)/torch.sum(adj==1)
        loss = loss/len(max_id_list)
        loss_all +=loss

    print('node_sim_loss: ',loss_all)
    return loss_all

def calculate_node_similarity(x,adj,if_neighbor=False):
    x_i = x.unsqueeze(1)  # Nx1xf
    x_j = torch.transpose(x_i, 0, 1)  # 1xNxf
    x_ij = x_i - x_j  # NxNxf
    x_ij = x_ij.float()

    ori_shape = x_ij.shape
    x_ij = torch.reshape(x_ij, (-1, ori_shape[-1]))  # (NxN)xf
    x_ij = torch.norm(x_ij, dim=1)

    x_ij = x_ij.reshape(ori_shape[:-1])  # NxN


    if if_neighbor:
        x_ij = x_ij * adj.float()  # NxN   #x_ij.max()=1.2
    else:
        x_ij = x_ij # NxN   #x_ij.max()=1.2
    return x_ij


def calculate_node_similarity_delta(x,adj,if_neighbor=False,delta=1.5):
    x_i = x.unsqueeze(1)  # Nx1xf
    x_j = torch.transpose(x_i, 0, 1)  # 1xNxf
    x_ij = x_i - x_j  # NxNxf
    x_ij = x_ij.float()

    ori_shape = x_ij.shape
    x_ij = torch.reshape(x_ij, (-1, ori_shape[-1]))  # (NxN)xf
    x_ij = torch.norm(x_ij, dim=1)

    x_ij = x_ij.reshape(ori_shape[:-1])  # NxN

    x_ij = (2 * delta - x_ij) / (2 * delta)
    x_ij = torch.clamp(x_ij, min=0) ** 2
    if if_neighbor:
        x_ij = x_ij * adj.float()  # NxN   #x_ij.max()=1.2
    else:
        x_ij = x_ij # NxN   #x_ij.max()=1.2


    return x_ij

def calculate_node_discrepancy(x,adj,if_neighbor=False):
    x_i = x.unsqueeze(1)  # Nx1xf
    x_j = torch.transpose(x_i, 0, 1)  # 1xNxf
    x_ij = x_i - x_j  # NxNxf
    x_ij = x_ij.float()

    if if_neighbor:
        x_ij = x_ij * adj.unsqueeze(2).float() # NxNxf
    else:
        x_ij = x_ij # NxNxf
    return x_ij


def calculate_mutual_node_similarity(h_list_T,h_list,edge_list,if_node=False,if_edge_discrepancy=False,if_edge_relation=False,if_neighbor=True,node_weight=1,edge_weight=1):
    "List: h_list_T: #[[NxC,Nxc....layers],[NxC,Nxc....layers],[NxC,Hxc....layers]...batches]"
    "List: edge_list: #[[edge_list,edge_list....layers],[edge_list,edge_list....layers],[edge_list,edge_list....layers]...batches]"
    loss_all = 0

    for h_list_T_b,h_list_b,edge_list_b in zip(h_list_T,h_list,edge_list):
        for x_T,x_S,edge in zip(h_list_T_b,h_list_b,edge_list_b):

            "tensor: X: NxF"
            "List: edge: [(1,2),(2,3)....]]"
            N = x_T.shape[0]
            adj = torch.zeros((N,N)).cuda()

            if if_neighbor:
                for (i,j) in edge:
                    if i>0:
                        adj[i-1,j-1] = 1
                        adj[i-1,j-1] = 1

            if if_node:
                dis_T_S = torch.norm(x_T-x_S, dim=1)
                loss_node = torch.mean(dis_T_S) * node_weight
                loss_all += loss_node
            if if_edge_relation:
                #L2 norm
                
                edge_T = calculate_node_similarity(x_T,adj,if_neighbor)# NxN
                edge_S = calculate_node_similarity(x_S,adj,if_neighbor)# NxN
                if if_neighbor:
                    loss_edge = torch.sum(torch.abs(edge_T-edge_S))/torch.sum(adj == 1)*edge_weight
                else:
                    loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / (x_T.shape[0])/ (x_T.shape[0])*edge_weight
                # loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / torch.sum(adj == 1)
                loss_all += loss_edge

            if if_edge_discrepancy:
                #L2 norm
                edge_T = calculate_node_discrepancy(x_T,adj,if_neighbor)# NxNxF
                edge_S = calculate_node_discrepancy(x_S,adj,if_neighbor)# NxNxF
                edge_discrepancy = edge_T-edge_S # NxNxF
                ori_shape = edge_discrepancy.shape
                edge_discrepancy = torch.reshape(edge_discrepancy, (-1, ori_shape[-1]))  # (NxN)xF
                edge_discrepancy = torch.norm(edge_discrepancy, dim=1)
                edge_discrepancy = edge_discrepancy.reshape(ori_shape[:-1])  # NxN
                # loss_edge = torch.sum(edge_discrepancy) / torch.sum(adj == 1)
                if if_neighbor:
                    loss_edge = torch.sum(edge_discrepancy)/torch.sum(adj == 1)
                else:
                    loss_edge = torch.sum(edge_discrepancy) / (x_T.shape[0])/ (x_T.shape[0])

                loss_all += loss_edge
    return loss_all/len(h_list_T)

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1) + 1e-8
def similarity(feat):
    feat = feat.float() #NxF
    tmp = L2(feat).detach() #Nx1
    feat = feat/tmp #(N,F) normalize
    return torch.einsum('ic,jc->ij', [feat, feat]) #N,N

def sim_dis_compute(f_S, f_T):#NxF
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[0]*f_T.shape[0])**2)
    sim_dis = sim_err.sum()
    return sim_dis


def calculate_node_similarity_cos(x,adj,if_neighbor=False):
    '''
    x:NXF

    '''

    x_ij = similarity(x)# NxN

    if if_neighbor:
        x_ij = x_ij * adj.float()  # NxN   #x_ij.max()=1.2
    else:
        x_ij = x_ij # NxN   #x_ij.max()=1.2
    return x_ij

def calculate_node_discrepancy_cos(x,adj,if_neighbor=False):
    x_i = x.unsqueeze(1)  # Nx1xf
    x_j = torch.transpose(x_i, 0, 1)  # 1xNxf
    x_ij = x_i - x_j  # NxNxf
    x_ij = x_ij.float()

    if if_neighbor:
        x_ij = x_ij * adj.unsqueeze(2).float() # NxNxf
    else:
        x_ij = x_ij # NxNxf
    return x_ij

def calculate_mutual_node_similarity_cos(h_list_T,h_list,edge_list,if_node=False,if_edge_discrepancy=False,if_edge_relation=False,if_neighbor=True):
    "List: h_list_T: #[[NxC,Nxc....layers],[NxC,Nxc....layers],[NxC,Hxc....layers]...batches]"
    "List: edge_list: #[[edge_list,edge_list....layers],[edge_list,edge_list....layers],[edge_list,edge_list....layers]...batches]"
    loss_all = 0

    for h_list_T_b,h_list_b,edge_list_b in zip(h_list_T,h_list,edge_list):
        for x_T,x_S,edge in zip(h_list_T_b,h_list_b,edge_list_b):

            "tensor: X: NxF"
            "List: edge: [(1,2),(2,3)....]]"
            N = x_T.shape[0]
            adj = torch.zeros((N,N)).cuda()

            if if_neighbor:
                for (i,j) in edge:
                    if i>0:
                        adj[i-1,j-1] = 1
                        adj[i-1,j-1] = 1

            if if_node:
                x_T = x_T.float()  # NxF
                x_S = x_S.float()  # NxF

                tmp_x_T = L2(x_T).detach()  # Nx1
                x_T = x_T / tmp_x_T  # (N,F) normalize

                tmp_x_S = L2(x_S).detach()  # Nx1
                x_S = x_S / tmp_x_S  # (N,F) normalize

                loss_node = torch.einsum('ic,ic->i', [x_T, x_S])  # N
                loss_node = torch.mean(loss_node)
                loss_all += loss_node
            if if_edge_relation:
                #L2 norm
                edge_T = calculate_node_similarity_cos(x_T,adj,if_neighbor)# NxN
                edge_S = calculate_node_similarity_cos(x_S,adj,if_neighbor)# NxN
                if if_neighbor:
                    # loss_edge = torch.sum((edge_T-edge_S)**2)/torch.sum(adj == 1)
                    loss_edge = torch.sum(torch.abs(edge_T - edge_S))/torch.sum(adj == 1)
                else:
                    loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / (x_T.shape[0]) / (x_T.shape[0])
                    # loss_edge = torch.sum((edge_T - edge_S)**2) / (x_T.shape[0])/ (x_T.shape[0])
                # loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / torch.sum(adj == 1)
                loss_all += loss_edge

            if if_edge_discrepancy:
                #L2 norm
                edge_T = calculate_node_discrepancy_cos(x_T,adj,if_neighbor)# NxNxF
                edge_S = calculate_node_discrepancy_cos(x_S,adj,if_neighbor)# NxNxF
                edge_discrepancy = edge_T-edge_S # NxNxF
                ori_shape = edge_discrepancy.shape
                edge_discrepancy = torch.reshape(edge_discrepancy, (-1, ori_shape[-1]))  # (NxN)xF
                edge_discrepancy = torch.norm(edge_discrepancy, dim=1)
                edge_discrepancy = edge_discrepancy.reshape(ori_shape[:-1])  # NxN
                # loss_edge = torch.sum(edge_discrepancy) / torch.sum(adj == 1)
                if if_neighbor:
                    loss_edge = torch.sum(edge_discrepancy)/torch.sum(adj == 1)
                else:
                    loss_edge = torch.sum(edge_discrepancy) / (x_T.shape[0])/ (x_T.shape[0])

                loss_all += loss_edge
    return loss_all/len(h_list_T)


def calculate_mutual_node_similarity_cos2(h_list_T,h_list,edge_list,if_node=False,if_edge_discrepancy=False,if_edge_relation=False,if_neighbor=True):
    "List: h_list_T: #[[NxC,Nxc....layers],[NxC,Nxc....layers],[NxC,Hxc....layers]...batches]"
    "List: edge_list: #[[edge_list,edge_list....layers],[edge_list,edge_list....layers],[edge_list,edge_list....layers]...batches]"
    loss_all = 0

    for h_list_T_b,h_list_b,edge_list_b in zip(h_list_T,h_list,edge_list):
        for x_T,x_S,edge in zip(h_list_T_b,h_list_b,edge_list_b):

            "tensor: X: NxF"
            "List: edge: [(1,2),(2,3)....]]"
            N = x_T.shape[0]
            adj = torch.zeros((N,N)).cuda()

            if if_neighbor:
                for (i,j) in edge:
                    if i>0:
                        adj[i-1,j-1] = 1
                        adj[i-1,j-1] = 1

            if if_node:
                x_T = x_T.float()  # NxF
                x_S = x_S.float()  # NxF

                dis_T_S = torch.norm(x_T-x_S, dim=1)
                loss_node = torch.mean(dis_T_S)
                loss_all += loss_node
            
            if if_edge_relation:
                #L2 norm
                edge_T = calculate_node_similarity_cos(x_T,adj,if_neighbor)# NxN
                edge_S = calculate_node_similarity_cos(x_S,adj,if_neighbor)# NxN
                if if_neighbor:
                    # loss_edge = torch.sum((edge_T-edge_S)**2)/torch.sum(adj == 1)
                    loss_edge = torch.sum(torch.abs(edge_T - edge_S))/torch.sum(adj == 1)
                else:
                    loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / (x_T.shape[0]) / (x_T.shape[0])
                    # loss_edge = torch.sum((edge_T - edge_S)**2) / (x_T.shape[0])/ (x_T.shape[0])
                # loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / torch.sum(adj == 1)
                loss_all += loss_edge

            if if_edge_discrepancy:
                #L2 norm
                edge_T = calculate_node_discrepancy_cos(x_T,adj,if_neighbor)# NxNxF
                edge_S = calculate_node_discrepancy_cos(x_S,adj,if_neighbor)# NxNxF
                edge_discrepancy = edge_T-edge_S # NxNxF
                ori_shape = edge_discrepancy.shape
                edge_discrepancy = torch.reshape(edge_discrepancy, (-1, ori_shape[-1]))  # (NxN)xF
                edge_discrepancy = torch.norm(edge_discrepancy, dim=1)
                edge_discrepancy = edge_discrepancy.reshape(ori_shape[:-1])  # NxN
                # loss_edge = torch.sum(edge_discrepancy) / torch.sum(adj == 1)
                if if_neighbor:
                    loss_edge = torch.sum(edge_discrepancy)/torch.sum(adj == 1)
                else:
                    loss_edge = torch.sum(edge_discrepancy) / (x_T.shape[0])/ (x_T.shape[0])

                loss_all += loss_edge
    return loss_all/len(h_list_T)


def calculate_mutual_node_similarity_cos3(h_list_T, h_list, edge_list, if_node=False, if_edge_discrepancy=False,
                                          if_edge_relation=False, if_neighbor=True):
    "List: h_list_T: #[[NxC,Nxc....layers],[NxC,Nxc....layers],[NxC,Hxc....layers]...batches]"
    "List: edge_list: #[[edge_list,edge_list....layers],[edge_list,edge_list....layers],[edge_list,edge_list....layers]...batches]"
    loss_all = 0

    for h_list_T_b, h_list_b, edge_list_b in zip(h_list_T, h_list, edge_list):
        for x_T, x_S, edge in zip(h_list_T_b, h_list_b, edge_list_b):

            "tensor: X: NxF"
            "List: edge: [(1,2),(2,3)....]]"
            N = x_T.shape[0]
            adj = torch.zeros((N, N)).cuda()

            if if_neighbor:
                for (i, j) in edge:
                    if i > 0:
                        adj[i - 1, j - 1] = 1
                        adj[i - 1, j - 1] = 1

            if if_node:
                x_T = x_T.float()  # NxF
                x_S = x_S.float()  # NxF

                dis_T_S = torch.norm(x_T - x_S, dim=1)
                loss_node = torch.mean(dis_T_S)
                loss_all += loss_node

            if if_edge_relation:
                # L2 norm
                edge_T = calculate_node_similarity_delta(x_T, adj, if_neighbor)  # NxN
                edge_S = calculate_node_similarity_delta(x_S, adj, if_neighbor)  # NxN
                if if_neighbor:
                    loss_edge = torch.sum((edge_T-edge_S)**2)/torch.sum(adj == 1)
                    # loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / torch.sum(adj == 1)
                else:
                    # loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / (x_T.shape[0]) / (x_T.shape[0])
                    loss_edge = torch.sum((edge_T - edge_S)**2) / (x_T.shape[0])/ (x_T.shape[0])
                # loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / torch.sum(adj == 1)
                loss_all += loss_edge

            if if_edge_discrepancy:
                # L2 norm
                edge_T = calculate_node_discrepancy_cos(x_T, adj, if_neighbor)  # NxNxF
                edge_S = calculate_node_discrepancy_cos(x_S, adj, if_neighbor)  # NxNxF
                edge_discrepancy = edge_T - edge_S  # NxNxF
                ori_shape = edge_discrepancy.shape
                edge_discrepancy = torch.reshape(edge_discrepancy, (-1, ori_shape[-1]))  # (NxN)xF
                edge_discrepancy = torch.norm(edge_discrepancy, dim=1)
                edge_discrepancy = edge_discrepancy.reshape(ori_shape[:-1])  # NxN
                # loss_edge = torch.sum(edge_discrepancy) / torch.sum(adj == 1)
                if if_neighbor:
                    loss_edge = torch.sum(edge_discrepancy) / torch.sum(adj == 1)
                else:
                    loss_edge = torch.sum(edge_discrepancy) / (x_T.shape[0]) / (x_T.shape[0])

                loss_all += loss_edge
    return loss_all / len(h_list_T)


if __name__ == "__main__":
    import h5py
    import torch
    spixel_path = r'E:\Code\Code_survey\Code_spix_embedding2\outputs\ID\0000.tif'
    segments = io.imread(spixel_path)
    # segments = torch.tensor(segments.astype(np.int64))
    segments = segments.astype(np.int64)[:256,:256][np.newaxis,...] #B,H,W
    inverse1, pack1 = np.unique(segments, return_inverse=True)
    pack1 = pack1.reshape(segments.shape)
    inverse1 = np.arange(0, inverse1.size)
    segments = inverse1[pack1]
    emb_path = r'E:\Code\Code_survey\Code_spix_embedding2\outputs\embedding\0001.hdf'
    with h5py.File(emb_path,'r') as f:
        embedding = f['main'][:]
    embedding = torch.tensor(embedding)[:,:256,:256].cuda().unsqueeze(0) #B,C,H,W
    #2,C,H,W
    embedding = torch.cat((embedding,embedding),dim=0)
    segments = np.vstack((segments,segments))
    print(embedding.shape,segments.shape)
    print('Number of spixs:',len(np.unique(segments)))
    # graph = get_graph_from_image(segments,embedding)
    h,adj,max_id_list = Segments2RAG(segments,embedding)
    #graph = Segments2RAG(torch.cat((segments.unsqueeze(0),segments.unsqueeze(0))), torch.cat((embedding.unsqueeze(0),embedding.unsqueeze(0))))
    print(h.shape,h.dtype,adj.shape,adj.dtype,max_id_list) #[0-276,277-..] 0 is 277

