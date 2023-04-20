from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
from torch.optim import lr_scheduler
import yaml
import time
import cv2
import h5py
import random
import logging
import argparse
import numpy as np
from PIL import Image
from attrdict import AttrDict
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_provider_celegans import Provider, Validation
from utils.show import show_affs, val_show,show_affs_emb,val_show_emd
from unet2d_residual import ResidualUNet2D_embedding as ResidualUNet2D_affs
from networks_emb import MobileNetV2,ESPNet,ERFNet,ENet,NestedUNet,PSPNet,Segformer,RAUNet
# from unet2d_residual_attention import ResidualUNet2D_embedding_attention as ResidualUNet2D_affs
from utils.utils import setup_seed
from loss.loss2 import WeightedMSE, WeightedBCE
from loss.loss2 import MSELoss, BCELoss, BCE_loss_func
from loss.loss_embedding_mse import embedding_loss
from loss.loss_discriminative import discriminative_loss
# from utils.evaluate import BestDice, AbsDiffFGLabels
from lib.evaluate.CVPPP_evaluate import BestDice, AbsDiffFGLabels, SymmetricBestDice
from utils.seg_mutex import seg_mutex
from utils.affinity_ours import multi_offset
from utils.emb2affs import embeddings_to_affinities
from postprocessing import merge_small_object, merge_func
from data.data_segmentation import relabel
from utils.cluster import cluster_ms
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from utils.metrics_bbbc import agg_jc_index, pixel_f1, remap_label, get_fast_pq
from utils.utils_rag_fc6 import construct_graph,calculate_self_node_similarity,calculate_mutual_node_similarity
from utils.cross_image_kd6 import cross_image_memory, calculate_CI_affinity_loss, calclulate_CI_graph_loss
import warnings

warnings.filterwarnings("ignore")


def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M',
            filename=path,
            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    # cfg.record_path = os.path.join(cfg.TRAIN.record_path, 'log')
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer


def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Validation(cfg, mode='validation')
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider


def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    stu_backbones = ['ENet','ESPNet','MobileNetV2']
    if cfg.MODEL.model_type in stu_backbones:
        model = eval(cfg.MODEL.model_type)().to(device)
    else:
        model = ResidualUNet2D_affs(in_channels=cfg.MODEL.input_nc,
                                    out_channels=cfg.MODEL.output_nc,
                                    nfeatures=cfg.MODEL.filters,
                                    emd=cfg.MODEL.emd,
                                    if_sigmoid=cfg.MODEL.if_sigmoid).to(device)
    if cfg.MODEL_T.model_type =='NestedUNet':
        model_T = eval(cfg.MODEL_T.model_type)().to(device)
    elif cfg.MODEL.model_type =='Segformer':
        model_T = Segformer(
        dims = (64, 128, 320, 512),      # dimensions of each stage
        heads = (1, 2, 5, 8),           # heads of each stage
        ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
        reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        num_layers = (3, 3, 18, 3),     # num layers of each stage
        decoder_dim = 256,              # decoder dimension
        num_classes = 16                 # number of segmentation classes
    ).to(device)
    else:
        model_T = ResidualUNet2D_affs(in_channels=cfg.MODEL_T.input_nc,
                                  out_channels=cfg.MODEL_T.output_nc,
                                  nfeatures=cfg.MODEL_T.filters,
                                  emd=cfg.MODEL_T.emd,
                                  if_sigmoid=cfg.MODEL_T.if_sigmoid).to(device)
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError(
                'Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    if cfg.TRAIN.if_KD:
        print('Load Teacher pretrained Model')
        model_path = os.path.join(cfg.TRAIN.model_T_path, 'model-%06d.ckpt' % cfg.TRAIN.model_T_id)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model_T.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        for k, v in model_T.named_parameters():
            v.requires_grad = False
    return model, model_T


def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0


def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters,
                                                                  cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(
                1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model, model_T, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    sum_loss_aff = 0.0
    sum_loss_affinity = 0.0
    sum_loss_graph = 0.0
    sum_loss_node = 0.0
    sum_loss_edge = 0.0
    sum_loss_CI_affinity = 0.0
    sum_loss_CI_graph = 0.0
    sum_loss_CI_node = 0.0
    sum_loss_CI_edge = 0.0
    sum_loss_mask = 0.0
    device = torch.device('cuda:0')
    offsets = multi_offset(list(cfg.DATA.shifts), neighbor=cfg.DATA.neighbor)

    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")
    criterion_dis = discriminative_loss
    criterion_mask = BCE_loss_func
    valid_mse = MSELoss()
    valid_bce = BCELoss()

    lr_strategies = ['steplr', 'multi_steplr', 'explr', 'lambdalr']
    if cfg.TRAIN.lr_mode == 'steplr':
        print('Step LR')
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.step_size, gamma=cfg.TRAIN.gamma)
    elif cfg.TRAIN.lr_mode == 'multi_steplr':
        print('Multi step LR')
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 150000],
                                                            gamma=cfg.TRAIN.gamma)
    elif cfg.TRAIN.lr_mode == 'explr':
        print('Exponential LR')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    elif cfg.TRAIN.lr_mode == 'lambdalr':
        print('Lambda LR')
        lambda_func = lambda epoch: (1.0 - epoch / cfg.TRAIN.total_iters) ** 0.9
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
    else:
        print('Other LR scheduler')

    memory_bank = cross_image_memory(memory_size=cfg.TRAIN.memory_size, contrast_size=cfg.TRAIN.contrast_size , t_channels=cfg.MODEL_T.emd, img_size = cfg.DATA.size).to(device)
    # params_list = nn.ModuleList([])
    # params_list.append(model_T)
    # params_list.append(memory_bank)
    # optimizer2 = torch.optim.Adam(params_list.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
    #                                  eps=0.01, weight_decay=1e-6, amsgrad=True)

    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        model_T.eval()

        iters += 1
        t1 = time.time()
        batch_data = train_provider.next()
        inputs = batch_data['img2'].cuda()
        target = batch_data['affs'].cuda()
        weightmap = batch_data['wmap'].cuda()
        target_ins = batch_data['seg'].cuda()
        affs_mask = batch_data['mask'].cuda()

        if cfg.TRAIN.lr_mode in lr_strategies:
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = cfg.TRAIN.base_lr

        optimizer.zero_grad()
        # optimizer2.zero_grad()
        if cfg.MODEL.model_type==2:
            embedding, _ = model(inputs)
        else:
            embedding = model(inputs)
            
        if cfg.MODEL_T.model_type==2:
            embedding_T, _ = model_T(inputs)
        else:
            embedding_T = model_T(inputs)

        ##############################
        # LOSS
        # loss = criterion(pred, target, weightmap)
    
        loss_aff, pred,_ = embedding_loss(embedding, target, weightmap, affs_mask, criterion, offsets,
                                              affs0_weight=cfg.TRAIN.dis_weight)
        _, pred_T,_ = embedding_loss(embedding_T, target, weightmap, affs_mask, criterion, offsets,
                                              affs0_weight=cfg.TRAIN.dis_weight)
        shift = 1
        # loss_aff_KD = criterion(pred, pred_T) * cfg.TRAIN.affinity_weight
        s1 = torch.prod(torch.tensor(pred.size()[-2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).cuda()
        loss_aff_KD = torch.sum((pred - pred_T) ** 2)/norm_term * cfg.TRAIN.affinity_weight
        batch = target_ins.shape[0]
        flag = 0
        flag2 = 0
        this_embedding, this_target_ins = memory_bank(embedding_T, target_ins)  # MxCxHxW  Mx1xHxW
        for b in range(batch):
            ins_number = torch.unique(target_ins[b])
            if len(ins_number)<=1:
                flag=1
        if flag==1:
            loss_graph = torch.tensor(0).cuda()
            loss_CI_graph = torch.tensor(0).cuda()
        else:
            h_list, edge_list = construct_graph(target_ins, [embedding], if_adjacent=cfg.TRAIN.if_neighbor)

            h_list_T, edge_list_T = construct_graph(target_ins, [embedding_T])
            t1 = time.time()
            # print('cost time of constructing graph:', t1-t0)

            # print('h_list_T',h_list_T[0][0].shape)
  
            loss_graph, loss_node, loss_edge = calculate_mutual_node_similarity(h_list_T, h_list, edge_list,
                                                        if_node=cfg.TRAIN.if_node,
                                                        if_edge_discrepancy=cfg.TRAIN.if_edge_discrepancy,
                                                        if_edge_relation=cfg.TRAIN.if_edge_relation,
                                                        if_neighbor=cfg.TRAIN.if_neighbor,
                                                        node_weight = cfg.TRAIN.node_weight,
                                                        edge_weight = cfg.TRAIN.edge_weight)

            #cros  image distillation
            M,C,H,W = this_embedding.shape
            this_embedding_new = torch.zeros((0,C,H,W)).cuda()
            this_target_ins_new = torch.zeros((0,1,H,W)).cuda()
            for m in range(M):
                ins_number = torch.unique(this_target_ins[m])
                if len(ins_number)>1:
                    this_embedding_new = torch.cat((this_embedding_new,this_embedding[m].unsqueeze(0)),dim=0)
                    this_target_ins_new = torch.cat((this_target_ins_new,this_target_ins[m].unsqueeze(0)),dim=0)
            if len(this_embedding_new) ==0:
                loss_CI_graph = torch.tensor(0).cuda()
            else:
                h_list_this, edge_list_this = construct_graph(this_target_ins_new, [this_embedding_new], if_adjacent=cfg.TRAIN.if_neighbor)
                # print('h_list_this',h_list_this[0][0].shape)
                loss_CI_graph, loss_CI_node, loss_CI_edge = calclulate_CI_graph_loss(h_list_this, h_list_T, h_list, 
                                                            if_node=cfg.TRAIN.if_node,
                                                            if_edge_discrepancy=cfg.TRAIN.if_edge_discrepancy,
                                                            if_edge_relation=cfg.TRAIN.if_edge_relation,
                                                            if_neighbor=cfg.TRAIN.if_neighbor,
                                                            node_weight = cfg.TRAIN.CI_node_weight,
                                                            edge_weight = cfg.TRAIN.CI_edge_weight, loss_type=cfg.TRAIN.cikd_loss_type)

        loss_CI_aff_KD = calculate_CI_affinity_loss(this_embedding_new,embedding, embedding_T, loss_type=cfg.TRAIN.cikd_loss_type)* cfg.TRAIN.CI_affinity_weight
        loss_CI_edge = (loss_CI_edge / cfg.TRAIN.contrast_size)*cfg.TRAIN.batch_size
        loss_CI_node = (loss_CI_node / cfg.TRAIN.contrast_size)*cfg.TRAIN.batch_size
        loss_CI_graph = (loss_CI_graph / cfg.TRAIN.contrast_size)*cfg.TRAIN.batch_size
        
        loss = loss_aff + loss_aff_KD + loss_graph + loss_CI_aff_KD + loss_CI_graph
        # loss = loss_aff +loss_graph
        loss.backward()
        # pred = F.relu(pred)
        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        # optimizer2.step()
        if cfg.TRAIN.lr_mode in lr_strategies:
            lr_scheduler.step()

        sum_loss += loss.item()
        sum_loss_aff += loss_aff.item()
        sum_loss_affinity += loss_aff_KD.item()
        sum_loss_graph += loss_graph.item()
        sum_loss_node += loss_node.item()
        sum_loss_edge += loss_edge.item()
        sum_loss_CI_affinity += loss_CI_aff_KD.item()
        sum_loss_CI_graph += loss_CI_graph.item()
        sum_loss_CI_node += loss_CI_node.item()
        sum_loss_CI_edge += loss_CI_edge.item()
        sum_loss_mask = 0.0
        # sum_loss_mask += loss_mask.item()
        sum_time += time.time() - t1

        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info(
                    'step %d, loss=%.6f, loss_aff=%.6f,loss_affinity=%.6f,l oss_graph=%.6f, loss_node=%.6f, loss_edge=%.6f, loss_CI_affinity=%6f, loss_CI_graph=%6f, loss_CI_node=%6f, loss_CI_edge=%6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                    % (iters, sum_loss, sum_loss_aff, sum_loss_affinity,sum_loss_graph,sum_loss_node, sum_loss_edge, sum_loss_CI_affinity, sum_loss_CI_graph, sum_loss_CI_node, sum_loss_CI_edge, current_lr, sum_time,
                       (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss, iters)

            else: 
                logging.info(
                    'step %d, loss=%.6f, loss_aff=%.6f,loss_affinity=%.6f,loss_graph=%.6f, loss_node=%.6f, loss_edge=%.6f, loss_CI_affinity=%6f, loss_CI_graph=%6f, loss_CI_node=%6f, loss_CI_edge=%6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)' \
                    % (iters, sum_loss / cfg.TRAIN.display_freq, \
                       sum_loss_aff / cfg.TRAIN.display_freq, \
                       sum_loss_affinity / cfg.TRAIN.display_freq, \
                       sum_loss_graph / cfg.TRAIN.display_freq, \
                       sum_loss_node / cfg.TRAIN.display_freq, \
                       sum_loss_edge / cfg.TRAIN.display_freq, \
                       sum_loss_CI_affinity/ cfg.TRAIN.display_freq,\
                       sum_loss_CI_graph/ cfg.TRAIN.display_freq,\
                        sum_loss_CI_node/ cfg.TRAIN.display_freq,\
                            sum_loss_CI_edge/ cfg.TRAIN.display_freq,\
                        current_lr, sum_time, \
                       (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                # logging.info('step %d, loss_dis=%.6f, loss_emd=%.6f' % (iters, loss_embedding_dis, loss_embedding))
                writer.add_scalar('sum_loss', sum_loss / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_aff', sum_loss_aff / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_affinity', sum_loss_affinity / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_graph', sum_loss_graph / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_node', sum_loss_node / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_edge', sum_loss_edge / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_CI_affinity', sum_loss_CI_affinity/ cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_CI_graph', sum_loss_CI_graph/ cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_CI_node', sum_loss_CI_node/ cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_CI_edge', sum_loss_CI_edge/ cfg.TRAIN.display_freq, iters)
            # f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq))
            f_loss_txt.write('step = %d, loss = %.6f, loss_aff=%.6f,l oss_affinity=%.6f,loss_graph=%.6f, loss_node=%.6f, loss_edge=%.6f, loss_CI_affinity=%6f, loss_CI_graph=%6f, loss_CI_node=%6f, loss_CI_edge=%6f ' % \
                             (iters, sum_loss / cfg.TRAIN.display_freq, sum_loss_aff / cfg.TRAIN.display_freq,
                              sum_loss_affinity / cfg.TRAIN.display_freq,sum_loss_graph / cfg.TRAIN.display_freq,sum_loss_node / cfg.TRAIN.display_freq,sum_loss_edge / cfg.TRAIN.display_freq, \
                                 sum_loss_CI_affinity/ cfg.TRAIN.display_freq,\
                       sum_loss_CI_graph/ cfg.TRAIN.display_freq,sum_loss_CI_node/ cfg.TRAIN.display_freq,\
                        sum_loss_CI_edge/ cfg.TRAIN.display_freq))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0.0
            sum_loss = 0.0
            sum_loss_aff = 0.0
            sum_loss_affinity = 0.0
            sum_loss_graph = 0.0
            sum_loss_node = 0.0
            sum_loss_edge = 0.0
            sum_loss_mask = 0.0
            sum_loss_CI_affinity = 0.0
            sum_loss_CI_graph = 0.0
            sum_loss_CI_node = 0.0
            sum_loss_CI_edge = 0.0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_affs(iters, batch_data['img1'], pred[:,-1], batch_data['affs'][:,-1], cfg.cache_path)
            val_show_emd(iters, batch_data['img1'][0], embedding[0], target_ins.squeeze().cpu().numpy()[0], target_ins.squeeze().cpu().numpy()[0], cfg.cache_path)

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                      'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_inpainting', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))


    print('*' * 20 + 'import data_provider_ours' + '*' * 20)

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model, model_T = build_model(cfg, writer)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                     eps=0.01, weight_decay=1e-6, amsgrad=True)
        # optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        # optimizer = optim.Adamax(model.parameters(), lr=cfg.TRAIN.base_l, eps=1e-8)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, model_T, nn.L1Loss(), optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')