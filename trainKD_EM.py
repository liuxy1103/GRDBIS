from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
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
from collections import OrderedDict
import multiprocessing as mp
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from utils_EM.utils_rag_fc import construct_graph,calculate_self_node_similarity,calculate_mutual_node_similarity
from utils_EM.cross_image_kd6 import cross_image_memory, calculate_CI_affinity_loss, calclulate_CI_graph_loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_provider_EM import Provider
from provider_valid_EM import Provider_valid
from loss_EM.loss import WeightedMSE, WeightedBCE
from loss_EM.loss import MSELoss, BCELoss
from utils_EM.show import show_affs, show_affs_whole
from unet3d_mala import UNet3D_MALA_embedding_small3 as UNet3D_MALA
from unet3d_mala import UNet3D_MALA_embedding
from utils_EM.utils import setup_seed, execute
from utils_EM.shift_channels import shift_func
from loss_EM.loss_embedding_mse import embedding_loss_norm1, embedding_loss_norm5, embedding_loss_norm_multi,inf_embedding_loss_norm1
import waterz
from utils_EM.lmc import mc_baseline
from utils_EM.fragment import watershed, randomlabel
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

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
        valid_provider = Provider_valid(cfg)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')

    print('load mala model!')
    model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc,
                        if_sigmoid=cfg.MODEL.if_sigmoid,
                        init_mode=cfg.MODEL.init_mode_mala,
                        show_feature=True,
                        emd=cfg.MODEL.emd).to(device)

    model_T = UNet3D_MALA_embedding(output_nc=cfg.MODEL.output_nc,
                        if_sigmoid=cfg.MODEL.if_sigmoid,
                        init_mode=cfg.MODEL.init_mode_mala,
                        show_feature=True,
                        emd=cfg.MODEL.emd).to(device)



    if cfg.MODEL.pre_train:
        print('Load pre-trained model ...')
        ckpt_path = os.path.join('../models2', \
            cfg.MODEL.trained_model_name, \
            'model-%06d.ckpt' % cfg.MODEL.trained_model_id)
        checkpoint = torch.load(ckpt_path)
        pretrained_dict = checkpoint['model_weights']
        if cfg.MODEL.trained_gpus > 1:
            pretained_model_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k[7:] # remove module.
                # name = k
                pretained_model_dict[name] = v
        else:
            pretained_model_dict = pretrained_dict

        from utils_EM.encoder_dict import ENCODER_DICT2, ENCODER_DECODER_DICT2
        model_dict = model.state_dict()
        encoder_dict = OrderedDict()
        if cfg.MODEL.if_skip == 'True':
            print('Load the parameters of encoder and decoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DECODER_DICT2}
        else:
            print('Load the parameters of encoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DICT2}
        model_dict.update(encoder_dict)
        model.load_state_dict(model_dict)

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
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
    else:
        return model

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
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model,model_T, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    sum_loss_aff = 0
    sum_loss_graph = 0
    sum_loss_node = 0
    sum_loss_edge = 0
    sum_loss_ci_node = 0
    sum_loss_ci_edge = 0
    sum_loss_ci_aff = 0
    sum_loss_ci_graph = 0
    sum_loss_aff_KD = 0
    device = torch.device('cuda:0')
    
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

    memory_bank = cross_image_memory(memory_size=cfg.TRAIN.memory_size, contrast_size=cfg.TRAIN.contrast_size , t_channels=cfg.MODEL.emd, img_size = 56).to(device)


    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        model_T.eval()
        iters += 1
        t1 = time.time()
        inputs, target, weightmap,target_ins = train_provider.next()
        
        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        optimizer.zero_grad()
        conv8, conv11, conv14, conv17, embedding = model(inputs)
        conv8_T, conv11_T, conv14_T, conv17_T, embedding_T = model_T(inputs)  # B,C,H,W
        ##############################


        # LOSS
        loss_aff, pred = embedding_loss_norm_multi(embedding, target, weightmap, criterion, affs0_weight=cfg.TRAIN.affs0_weight,shift=cfg.DATA.shift_channels)
        shift = 1
        
        t0 = time.time()
        target_ins = target_ins.unsqueeze(1)
        loss_graph = torch.zeros(1).cuda()
        loss_node = torch.zeros(1).cuda()
        loss_edge = torch.zeros(1).cuda()
        loss_aff_KD = torch.zeros(1).to(device)
        assert len(target_ins.shape)==5

        for emb,emb_T in zip([embedding,conv17,conv14],[embedding_T,conv17_T,conv14_T]):
            if emb_T.shape != emb.shape:
                transform = nn.Conv3d(emb.shape[1], emb_T.shape[1], 1, bias=False).to(device)
                transform.weight.data.uniform_(-0.005, 0.005)
                emb = transform(emb)
            h_list, edge_list = construct_graph(target_ins, [emb], if_adjacent=cfg.TRAIN.if_neighbor)
            h_list_T, edge_list_T = construct_graph(target_ins, [emb_T])
            t1 = time.time()
            # print('cost time of constructing graph:', t1-t0)
            # print(h_list,edge_list)
            loss_graph_tmp, loss_node_tmp, loss_edge_tmp = calculate_mutual_node_similarity(h_list_T, h_list, edge_list,
                                                              if_node=cfg.TRAIN.if_node,
                                                              if_edge_discrepancy=cfg.TRAIN.if_edge_discrepancy,
                                                              if_edge_relation=cfg.TRAIN.if_edge_relation,
                                                              node_weight=cfg.TRAIN.node_weight,
                                                              edge_weight=cfg.TRAIN.edge_weight)
   
            loss_graph += loss_graph_tmp
            loss_node += loss_node_tmp
            loss_edge += loss_edge_tmp
            pred_tmp = inf_embedding_loss_norm1(emb)
            pred_T_tmp = inf_embedding_loss_norm1(emb_T)
            loss_aff_KD_tmp = criterion(pred_tmp, pred_T_tmp) * cfg.TRAIN.affinity_weight
            loss_aff_KD += loss_aff_KD_tmp


        #cros  image distillation
        this_embedding, this_target_ins = memory_bank(embedding_T, target_ins.cuda())  # MxZxCxHxW  Mx1xZxHxW
        # print('this_embedding shape:', this_embedding.shape)
        h_list_this, edge_list_this = construct_graph(this_target_ins, [this_embedding], if_adjacent=cfg.TRAIN.if_neighbor)
        h_list_T, edge_list_T = construct_graph(target_ins, [embedding_T])
        h_list, edge_list = construct_graph(target_ins, [embedding], if_adjacent=cfg.TRAIN.if_neighbor)
        loss_CI_aff_KD = calculate_CI_affinity_loss(this_embedding,embedding, embedding_T, loss_type=cfg.TRAIN.cikd_loss_type)* cfg.TRAIN.CI_affinity_weight
        loss_CI_graph, loss_CI_node, loss_CI_edge = calclulate_CI_graph_loss(h_list_this, h_list_T, h_list, 
                                                      if_node=cfg.TRAIN.if_node,
                                                      if_edge_discrepancy=cfg.TRAIN.if_edge_discrepancy,
                                                      if_edge_relation=cfg.TRAIN.if_edge_relation,
                                                      if_neighbor=cfg.TRAIN.if_neighbor,
                                                      node_weight = cfg.TRAIN.CI_node_weight,
                                                      edge_weight = cfg.TRAIN.CI_edge_weight, loss_type=cfg.TRAIN.cikd_loss_type) 
        loss_CI_edge = (loss_CI_edge / cfg.TRAIN.contrast_size)*cfg.TRAIN.batch_size
        loss_CI_node = (loss_CI_node / cfg.TRAIN.contrast_size)*cfg.TRAIN.batch_size
        loss_CI_graph = (loss_CI_graph / cfg.TRAIN.contrast_size)*cfg.TRAIN.batch_size
        loss_total = loss_aff + loss_aff_KD + loss_graph + loss_CI_aff_KD + loss_CI_graph
        loss_total.backward()


        pred[:, 1, :, :shift, :] = pred[:, 1, :, shift:shift*2, :]
        pred[:, 2, :, :, :shift] = pred[:, 2, :, :, shift:shift*2]
        pred[:, 0, :shift, :, :] = pred[:, 0, shift:shift*2, :, :]
        pred = F.relu(pred)
        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        
        sum_loss += loss_total.item()
        sum_loss_aff += loss_aff.item()
        sum_loss_node += loss_node.item()
        sum_loss_edge += loss_edge.item()
        sum_loss_graph += loss_graph.item()
        sum_loss_ci_edge += loss_CI_edge.item()
        sum_loss_ci_node += loss_CI_node.item()
        sum_loss_ci_graph += loss_CI_graph.item()
        sum_loss_ci_aff += loss_CI_aff_KD.item()
        sum_time += time.time() - t1
        sum_loss_aff_KD += loss_aff_KD.item()
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss = %.6f, sum_loss_aff = %.6f, sum_loss_aff_KD = %.6f, sum_loss_graph = %.6f,sum_loss_node = %.6f,sum_loss_edge = %.6f, \
                     sum_loss_ci_aff = %.6f, sum_loss_ci_graph = %.6f, sum_loss_ci_node = %.6f, sum_loss_ci_edge = %.6f, (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss * 1,sum_loss_aff,sum_loss_aff_KD,sum_loss_graph,sum_loss_node,sum_loss_edge,\
                                sum_loss_ci_aff, sum_loss_ci_graph, sum_loss_ci_node, sum_loss_ci_edge, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info('step %d, loss = %.6f, sum_loss_aff = %.6f, sum_loss_aff_KD = %.6f, sum_loss_graph = %.6f, sum_loss_node = %.6f,sum_loss_edge = %.6f,\
                    sum_loss_ci_aff = %.6f, sum_loss_ci_graph = %.6f, sum_loss_ci_node = %.6f, sum_loss_ci_edge = %.6f, (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss / cfg.TRAIN.display_freq, sum_loss_aff / cfg.TRAIN.display_freq,
                               sum_loss_aff_KD/ cfg.TRAIN.display_freq,
                               sum_loss_graph / cfg.TRAIN.display_freq,
                               sum_loss_node / cfg.TRAIN.display_freq,
                               sum_loss_edge / cfg.TRAIN.display_freq,
                               sum_loss_ci_aff / cfg.TRAIN.display_freq,
                               sum_loss_ci_graph / cfg.TRAIN.display_freq,
                               sum_loss_ci_node / cfg.TRAIN.display_freq,
                               sum_loss_ci_edge / cfg.TRAIN.display_freq, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step %d, loss = %.6f, sum_loss_aff = %.6f, sum_loss_aff_KD = %.6f, sum_loss_graph = %.6f, sum_loss_node = %.6f,sum_loss_edge = %.6f\
                sum_loss_ci_aff = %.6f, sum_loss_ci_graph = %.6f, sum_loss_ci_node = %.6f, sum_loss_ci_edge = %.6f'
                            % (iters, sum_loss / cfg.TRAIN.display_freq, sum_loss_aff / cfg.TRAIN.display_freq,
                               sum_loss_aff_KD/ cfg.TRAIN.display_freq,
                               sum_loss_graph / cfg.TRAIN.display_freq,
                               sum_loss_node / cfg.TRAIN.display_freq,
                               sum_loss_edge / cfg.TRAIN.display_freq,
                               sum_loss_ci_aff / cfg.TRAIN.display_freq,
                               sum_loss_ci_graph / cfg.TRAIN.display_freq,
                               sum_loss_ci_node / cfg.TRAIN.display_freq,
                               sum_loss_ci_edge / cfg.TRAIN.display_freq))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0
            sum_loss_aff = 0
            sum_loss_graph = 0
            sum_loss_edge = 0
            sum_loss_node = 0
            sum_loss_aff_KD = 0
            sum_loss_ci_aff = 0
            sum_loss_ci_edge = 0
            sum_loss_ci_node = 0
            sum_loss_ci_graph = 0
        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_affs(iters, inputs, pred[:,:3], target[:,:3], cfg.cache_path, model_type=cfg.MODEL.model_type)

        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []
                for k, batch in enumerate(dataloader, 0):
                    inputs, target, weightmap = batch
                    inputs = inputs.cuda()
                    target = target.cuda()
                    weightmap = weightmap.cuda()
                    with torch.no_grad():
                        conv8, conv11, conv14, conv17, embedding = model(inputs)

                    tmp_loss, pred = embedding_loss_norm_multi(embedding, target, weightmap, criterion, affs0_weight=cfg.TRAIN.affs0_weight,shift=cfg.DATA.shift_channels)

                    shift = 1
                    pred[:, 1, :, :shift, :] = pred[:, 1, :, shift:shift*2, :]
                    pred[:, 2, :, :, :shift] = pred[:, 2, :, :, shift:shift*2]
                    pred[:, 0, :shift, :, :] = pred[:, 0, shift:shift*2, :, :]
                    pred = F.relu(pred)
                    losses_valid.append(tmp_loss.item())
                    valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
                epoch_loss = sum(losses_valid) / len(losses_valid)
                out_affs = valid_provider.get_results()
                gt_affs = valid_provider.get_gt_affs().copy()
                gt_seg = valid_provider.get_gt_lb()
                valid_provider.reset_output()
                out_affs = out_affs[:3]
                # gt_affs = gt_affs[:, :3]
                show_affs_whole(iters, out_affs, gt_affs, cfg.valid_path)

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
