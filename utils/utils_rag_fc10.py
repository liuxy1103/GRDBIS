# -*- coding: utf-8 -*-
# @Time : 2021/12/3
# @Author : Xiaoyu Liu
# @Email : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm
#Function: construct fc rag for instances

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
    loss_node_all = 0
    loss_edge_all = 0

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
                loss_node_all += loss_node
            if if_edge_relation:
                #L2 nor
                # x_T = F.normalize(x_T, p=2, dim=1)
                # x_S = F.normalize(x_S, p=2, dim=1)
                edge_T = torch.mm(x_T, x_T.transpose(0, 1))  #(N,F)x(F,N)
                edge_S = torch.mm(x_S, x_S.transpose(0, 1))  #(N,F)x(F,N)
                dis_edge = edge_T - edge_S
                norm = torch.norm(dis_edge.reshape(-1),p=2)
                dis_edge = dis_edge/norm
                if if_neighbor:
                    loss_edge = torch.sum(torch.abs(dis_edge)**2)/torch.sum(adj == 1)*edge_weight
                else:
                    loss_edge = torch.sum(torch.abs(dis_edge)**2) / (x_T.shape[0])/ (x_T.shape[0])*edge_weight
                # loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / torch.sum(adj == 1)
                loss_all += loss_edge
                loss_edge_all += loss_edge
            if if_edge_discrepancy:  #False
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
                loss_edge_all += loss_edge

    return loss_all/len(h_list_T), loss_node_all/len(h_list_T), loss_edge_all/len(h_list_T)

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

# -*- coding: utf-8 -*-
# @Time : 2021/12/3
# @Author : Xiaoyu Liu
# @Email : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm
#Function: construct fc rag for instances

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
    loss_node_all = 0
    loss_edge_all = 0

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
                loss_node_all += loss_node
            if if_edge_relation:
                #L2 nor
                # x_T = F.normalize(x_T, p=2, dim=1)
                # x_S = F.normalize(x_S, p=2, dim=1)
                edge_T = torch.mm(x_T, x_T.transpose(0, 1))  #(N,F)x(F,N)
                edge_S = torch.mm(x_S, x_S.transpose(0, 1))  #(N,F)x(F,N)
                dis_edge = edge_T - edge_S
                norm = torch.norm(dis_edge.reshape(-1),p=2)
                dis_edge = dis_edge/norm
                if if_neighbor:
                    loss_edge = torch.sum(torch.abs(dis_edge)**2)/torch.sum(adj == 1)*edge_weight
                else:
                    loss_edge = torch.sum(torch.abs(dis_edge)**2) / (x_T.shape[0])/ (x_T.shape[0])*edge_weight
                # loss_edge = torch.sum(torch.abs(edge_T - edge_S)) / torch.sum(adj == 1)
                loss_all += loss_edge
                loss_edge_all += loss_edge
            if if_edge_discrepancy:  #False
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
                loss_edge_all += loss_edge

    return loss_all/len(h_list_T), loss_node_all/len(h_list_T), loss_edge_all/len(h_list_T)

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

