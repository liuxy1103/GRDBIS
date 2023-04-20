import os
import cv2
import h5py
from torch.functional import split
import yaml
import time
import argparse
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from attrdict import AttrDict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.loss_embedding_mse import embedding_loss,embedding2affs
from data_provider_cvppp import Provider, Validation
from utils.show import show_affs, val_show, val_show_emd, val_show_emd_layers
from utils.utils import setup_seed
from loss.loss import WeightedMSE, WeightedBCE
from loss.loss import MSELoss, BCELoss, BCE_loss_func
from loss.loss_discriminative import discriminative_loss
from unet2d_residual import ResidualUNet2D_embedding as ResidualUNet2D_affs

from networks_emb import MobileNetV2,ESPNet,ERFNet,ENet,NestedUNet,PSPNet,Segformer,RAUNet

# from utils.evaluate import BestDice, AbsDiffFGLabels, SymmetricBestDice
from lib.evaluate.CVPPP_evaluate import BestDice,DiffFGLabels, AbsDiffFGLabels, SymmetricBestDice, SymmetricBestDice_max
from utils.seg_mutex import seg_mutex
from utils.affinity_ours import multi_offset
from utils.emb2affs import embeddings_to_affinities
# from postprocessing import merge_small_object
from data.data_segmentation import relabel
from utils.cluster import cluster_ms  # , cluster_hdbscan, cluster_consistency
from skimage import io
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import time
import warnings

warnings.filterwarnings("ignore")


def merge_small_object(seg, threshold=5, window=5):
    uid, uc = np.unique(seg, return_counts=True)
    for (ids, size) in zip(uid, uc):
        if size > threshold:
            continue
        # print(seg.shape)
        # print(ids)
        # print(np.where(seg == ids))
        pos_x, pos_y = np.where(seg == ids)
        pos_x = int(np.sum(pos_x) // np.size(pos_x))
        pos_y = int(np.sum(pos_y) // np.size(pos_y))
        pos_x = pos_x - window // 2
        pos_y = pos_y - window // 2
        seg_crop = seg[pos_x:pos_x+window, pos_y:pos_y+window]
        temp_uid, temp_uc = np.unique(seg_crop, return_counts=True)
        rank = np.argsort(-temp_uc)
        if len(temp_uc) > 2:
            if temp_uid[rank[0]] == 0:
                if temp_uid[rank[1]] == ids:
                    max_ids = temp_uid[rank[2]]
                else:
                    max_ids = temp_uid[rank[1]]
            else:
                max_ids = temp_uid[rank[0]]
            seg[seg==ids] = max_ids
    return seg

def merge_func(seg, step=4):
    seg = merge_small_object(seg)
    seg = merge_small_object(seg, threshold=20, window=11)
    seg = merge_small_object(seg, threshold=50, window=11)
    seg = merge_small_object(seg, threshold=300, window=21)
    return seg

def merge_func2(seg, step=4):
    seg = merge_small_object(seg)
    seg = merge_small_object(seg, threshold=20, window=11)
    seg = merge_small_object(seg, threshold=50, window=11)
    # seg = merge_small_object(seg, threshold=100, window=21)
    seg = merge_small_object(seg, threshold=500, window=101)
    return seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='baseline_emb16_mutex_MobileNetV2', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default='2021-11-10--08-14-44_baeline_embedding16_mutex_T2')
    parser.add_argument('-id', '--model_id', type=int, default=0)
    parser.add_argument('-m', '--mode', type=str, default='validation')  # test validation
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('-sw', '--show', action='store_false', default=True)#
    parser.add_argument('-swa', '--show_affinity', action='store_false', default=True)#
    parser.add_argument('-se', '--show_embedding', action='store_true', default=False)
    parser.add_argument('-sel', '--show_embedding_layers', action='store_true', default=False)
    parser.add_argument('-ne', '--norm_embedding', action='store_true', default=True)
    parser.add_argument('-sn', '--show_num', type=int, default=10)
    parser.add_argument('-sd', '--stride', type=int, default=None)
    parser.add_argument('-nb', '--neighbor', type=int, default=None)
    parser.add_argument('-sf', '--shifts', type=str, default=None)
    parser.add_argument('-sbd', '--if_sbd', action='store_false', default=True)
    parser.add_argument('-es', '--eps', type=float, default=0.5)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    out_path = os.path.join('../inference_extension', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_' + str(args.model_id)
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)
    affs_img_path = os.path.join(out_affs, 'affs_img')
    seg_img_path = os.path.join(out_affs, 'seg_img')
    if not os.path.exists(affs_img_path):
        os.makedirs(affs_img_path)
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)

    device = torch.device('cuda:0')

    model = ResidualUNet2D_affs(in_channels=cfg.MODEL.input_nc,
                                out_channels=cfg.MODEL.output_nc,
                                nfeatures=cfg.MODEL.filters,
                                emd=cfg.MODEL.emd,
                                if_sigmoid=cfg.MODEL.if_sigmoid,
                                show_feature=True).to(device)

    stu_backbones = ['ENet','ESPNet','MobileNetV2']
    if cfg.MODEL.model_type in stu_backbones:
        model = eval(cfg.MODEL.model_type)().to(device)
    else:
        model = ResidualUNet2D_affs(in_channels=cfg.MODEL.input_nc,
                                    out_channels=cfg.MODEL.output_nc,
                                    nfeatures=cfg.MODEL.filters,
                                    emd=cfg.MODEL.emd,
                                    if_sigmoid=cfg.MODEL.if_sigmoid).to(device)
    if cfg.MODEL.model_type =='NestedUNet':
        model = eval(cfg.MODEL.model_type)().to(device)
    model_type =2
    print('Used Model type:',cfg.MODEL.model_type)


    ckpt_path = os.path.join('../models_extension', trained_model, 'stu_KD.ckpt')
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        # name = k[7:] # remove module.
        name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    valid_provider = Validation(cfg, mode=args.mode)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True)


    criterion = MSELoss()
    criterion_mask = BCE_loss_func

    if args.shifts is None:
        shifts = list(cfg.DATA.shifts)
    else:
        shifts_str = args.shifts
        split = shifts_str.split(',')
        shifts = []
        for k in split:
            shifts.append(int(k))
    if args.neighbor is None:
        neighbor = cfg.DATA.neighbor
    else:
        neighbor = args.neighbor
    print('shifts is', shifts, end=' ')
    print('neighbor is', neighbor)
    offsets = multi_offset(list(cfg.DATA.shifts), neighbor=cfg.DATA.neighbor)
    losses_valid = []
    dice = []
    dice_max = []
    diff = []
    diff2 = []
    all_voi = []
    all_arand = []
    affs = []
    seg = []
    if args.stride is None:
        stride = list(cfg.DATA.strides)
    else:
        stride = [args.stride, args.stride]

    start_time = time.time()
    time_total = 0
    f_txt = open(os.path.join(out_affs, 'score.txt'), 'w')
    for k, batch in enumerate(val_loader, 0):

        batch_data = batch
        inputs = batch_data['image'].cuda()
        target = batch_data['affs'].cuda()
        weightmap = batch_data['wmap'].cuda()
        target_ins = batch_data['seg'].cuda()
        affs_mask = batch_data['mask'].cuda()

        with torch.no_grad():
            torch.cuda.synchronize()
            time_s = time.time()
            if cfg.MODEL.model_type == 2:
                embedding, _= model(inputs)
            else:
                embedding= model(inputs)
            
            torch.cuda.synchronize()
            time_e = time.time()
            time_per = time_e - time_s
        if args.mode == 'test':
            losses_valid.append(0.0)
            pred = embedding2affs(embedding,offsets)
        else:
            loss_embedding, pred, _ = embedding_loss(embedding, target, weightmap, affs_mask, criterion,
                                                     offsets, affs0_weight=cfg.TRAIN.dis_weight)

            tmp_loss = loss_embedding
            losses_valid.append(tmp_loss.item())

        time_total += time_per
        output_affs = np.squeeze(pred.data.cpu().numpy())
        affs.append(output_affs.copy())
        # post-processing
        gt_ins = np.squeeze(batch_data['seg'].numpy()).astype(np.uint8)
        gt_mask = gt_ins.copy()
        gt_mask[gt_mask != 0] = 1
        pred_seg = seg_mutex(output_affs, offsets=offsets, strides=list(cfg.DATA.strides), mask=gt_mask).astype(np.uint16)
        pred_seg = merge_func(pred_seg)
        pred_seg = relabel(pred_seg)
        seg.append(pred_seg.copy())
        pred_seg = pred_seg.astype(np.uint16)
        gt_ins = gt_ins.astype(np.uint16)

        # evaluate
        if args.mode == 'test':
            temp_dice = 0.0
            temp_diff = np.max(pred_seg)
            temp_diff2 = np.max(pred_seg)
            arand = 0.0
            voi_sum = 0.0
            temp_dice_max = 0.0
        else:
            if args.if_sbd:
                temp_dice = SymmetricBestDice(pred_seg, gt_ins)
                temp_dice_max = SymmetricBestDice_max(pred_seg, gt_ins)
            else:
                temp_dice = 0.0
                temp_dice_max = 0.0
            temp_diff = AbsDiffFGLabels(pred_seg, gt_ins)
            temp_diff2 = DiffFGLabels(pred_seg, gt_ins)
            arand = adapted_rand_ref(gt_ins, pred_seg, ignore_labels=(0))[0]
            voi_split, voi_merge = voi_ref(gt_ins, pred_seg, ignore_labels=(0))
            voi_sum = voi_split + voi_merge
        print('image=%d, SBD=%.6f, DiC=%.6f, DiC2=%.6f, VOI=%.6f, ARAND=%.6f' % (k, temp_dice, temp_diff, temp_diff2, voi_sum, arand))
        f_txt.write('image=%d, SBD=%.6f, DiC=%.6f, DiC2=%.6f, VOI=%.6f, ARAND=%.6f' % (k, temp_dice, temp_diff, temp_diff2, voi_sum, arand))
        f_txt.write('\n')
        all_voi.append(voi_sum)
        all_arand.append(arand)
        dice.append(temp_dice)
        dice_max.append(temp_dice_max)
        diff.append(temp_diff)
        diff2.append(temp_diff2)
        affs_gt = batch_data['affs'].numpy()

    cost_time = time.time() - start_time
    print('Inference Cost Time:', time_total)
    epoch_loss = sum(losses_valid) / len(losses_valid)
    sbd = sum(dice) / len(dice)
    sbd_max = sum(dice_max) / len(dice_max)
    # sbd = 0.0
    dic = sum(diff) / len(diff)
    dic2 = sum(diff2) / len(diff2)
    mean_voi = sum(all_voi) / len(all_voi)
    mean_arand = sum(all_arand) / len(all_arand)
    print('model-%d, valid-loss=%.6f, SBD_min=%.6f, SBD_max=%.6f, DiC=%.6f,DiC2=%.6f, VOI=%.6f, ARAND=%.6f' % \
          (args.model_id, epoch_loss, sbd, sbd_max, dic,dic2, mean_voi, mean_arand), flush=True)
    print('COST TIME = %.6f' % cost_time)

    f_txt.write('model-%d, valid-loss=%.6f, SBD_min=%.6f, SBD_max=%.6f, DiC=%.6f, DiC2=%.6f, VOI=%.6f, ARAND=%.6f' % \
                (args.model_id, epoch_loss, sbd, sbd_max, dic,dic2, mean_voi, mean_arand))
    f_txt.write('\n')
    f_txt.close()