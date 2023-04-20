from multiprocessing.spawn import import_main_path
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
from data_provider_bbbc import Provider, Validation
from unet2d_residual import ResidualUNet2D_embedding as ResidualUNet2D_affs
from networks_emb import MobileNetV2,ESPNet,ERFNet,ENet,NestedUNet,PSPNet,Segformer,RAUNet
from utils.show import show_affs, val_show, show_affs_emb, val_show_emd, show_affs_all
from utils.utils import setup_seed
from loss.loss import WeightedMSE, WeightedBCE
from loss.loss import MSELoss, BCELoss, BCE_loss_func
from utils.metrics_bbbc import agg_jc_index, pixel_f1, remap_label, get_fast_pq
from utils.seg_mutex import seg_mutex
from utils.affinity_ours import multi_offset
from postprocessing import merge_small_object
from data.data_segmentation import relabel
from utils.cluster import cluster_ms
from utils.emb2affs import embeddings_to_affinities
from loss.loss_embedding_mse import embedding_loss
from sklearn.decomposition import PCA
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import hdbscan
import warnings
warnings.filterwarnings("ignore")

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
    seg = merge_small_object(seg, threshold=500, window=101)
    return seg

def cluster(emb, clustering_alg, semantic_mask=None):
    output_shape = emb.shape[1:]
    flattened_embeddings = emb.reshape(emb.shape[0], -1).transpose()

    result = np.zeros(flattened_embeddings.shape[0])

    if semantic_mask is not None:
        flattened_mask = semantic_mask.reshape(-1)
        assert flattened_mask.shape[0] == flattened_embeddings.shape[0]
    else:
        flattened_mask = np.ones(flattened_embeddings.shape[0])

    if flattened_mask.sum() == 0:
        # return zeros for empty masks
        return result.reshape(output_shape)

    # cluster only within the foreground mask
    clusters = clustering_alg.fit_predict(flattened_embeddings[flattened_mask == 1])
    # always increase the labels by 1 cause clustering results start from 0 and we may loose one object
    result[flattened_mask == 1] = clusters + 1

    return result.reshape(output_shape)

def cluster_hdbscan(emb, min_size, eps, min_samples=None, semantic_mask=None):
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_size, cluster_selection_epsilon=eps, min_samples=min_samples)
    return cluster(emb, clustering, semantic_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='baseline_emb16_mutex_T2', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default='2022-12-03--10-55-43_baseline_emb16_mutex_T2')
    parser.add_argument('-id', '--model_id', type=int, default=0)
    parser.add_argument('-m', '--mode', type=str, default='validation') # test,validation
    parser.add_argument('-pt', '--post_type', type=str, default='Mutex') # HDBSCAN Mutex Meanshift
    parser.add_argument('-ti', '--set_thres_if', action='store_true', default=True) # HDBSCAN Meanshift
    parser.add_argument('-thres', '--set_thres', type=float, default=0.2) # HDBSCAN
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('-sw', '--show', action='store_false', default=True)
    parser.add_argument('-se', '--show_embedding', action='store_true', default=True)
    parser.add_argument('-ne', '--norm_embedding', action='store_true', default=False)
    parser.add_argument('-sn', '--show_num', type=int, default=1)
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

    if 'embedding' in args.cfg:
        if_embedding = True
    else:
        if_embedding = False

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    out_path = os.path.join('../inference', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'embs_'+str(args.model_id)
    out_embs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_embs):
        os.makedirs(out_embs)
    print('out_path: ' + out_embs)
    embs_img_path = os.path.join(out_embs, 'embs_img')
    affs_img_path = os.path.join(out_embs, 'affs_img')
    entropy_img_path = os.path.join(out_embs, 'entropy_img')
    seg_img_path = os.path.join(out_embs, 'seg_img')
    if not os.path.exists(embs_img_path):
        os.makedirs(embs_img_path)
    if not os.path.exists(affs_img_path):
        os.makedirs(affs_img_path)
    if not os.path.exists(entropy_img_path):
        os.makedirs(entropy_img_path)
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)

    device = torch.device('cuda:0')
    try:
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

        print('Used Model type:',cfg.MODEL.model_type)
    except:
        model = ResidualUNet2D_affs(in_channels=cfg.MODEL.input_nc,
                                    out_channels=cfg.MODEL.output_nc,
                                    nfeatures=cfg.MODEL.filters,
                                    emd=cfg.MODEL.emd,
                                    if_sigmoid=cfg.MODEL.if_sigmoid).to(device)

    ckpt_path = os.path.join('../output/models', trained_model, 'stu_KD.ckpt')
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    valid_provider = Validation(cfg, mode=args.mode)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                            shuffle=False, drop_last=False, pin_memory=True)

    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    elif cfg.TRAIN.loss_func == 'others':
        criterion = create_loss(cfg.Loss.delta_var, cfg.Loss.delta_dist, cfg.Loss.alpha, cfg.Loss.beta,
                                cfg.Loss.gamma, cfg.Loss.unlabeled_push, cfg.Loss.instance_weight,
                                cfg.Loss.consistency_weight, cfg.Loss.kernel_threshold, cfg.Loss.instance_loss, cfg.Loss.spoco)
    else:
        raise AttributeError("NO this criterion")

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
    offsets = multi_offset(shifts, neighbor=neighbor)
    losses_valid = []
    dice = []
    dice_max = []
    bd = []
    diff = []
    all_voi = []
    all_arand = []
    embs = []
    seg = []
    if args.stride is None:
        stride = list(cfg.DATA.strides)
    else:
        stride = [args.stride, args.stride]

    start_time = time.time()
    f_txt = open(os.path.join(out_embs, 'score.txt'), 'w')

    aji_score = []
    dice_score = []
    f1_score = []
    pq_score = []

    for k, batch in enumerate(val_loader, 0):
        # if k != 6:
        #     continue
        batch_data = batch
        im_f = batch_data['img1'].cuda()
        im_g = batch_data['img2'].cuda()
        target = batch_data['affs'].cuda()
        weightmap = batch_data['wmap'].cuda()
        target_ins = batch_data['seg'].cuda().float()
        affs_mask = batch_data['mask'].cuda().float()

        with torch.no_grad():
            emb_f, _ = model(im_f)

        losses_valid.append(0.0)

        gt_ins = np.squeeze(target_ins.cpu().numpy()).astype(np.uint8)
        gt_mask = gt_ins.copy()
   
        gt_mask[gt_mask != 0] = 1

        affs_gt = np.squeeze(target.cpu().numpy())
        loss_embedding, pred,_ = embedding_loss(emb_f, target, weightmap, affs_mask, criterion, offsets)

        pred_tmp = F.relu(pred)
        entropy = -(pred_tmp * torch.log(pred_tmp + 1e-10)+ (1-pred_tmp)*torch.log(1-pred_tmp+ 1e-10))
        entropy_out = entropy[0].unsqueeze(0)
        entropy_out = np.squeeze(entropy_out.data.cpu().numpy())

        out_affs = np.squeeze(pred.data.cpu().numpy())
        pred_seg = seg_mutex(out_affs, offsets=offsets, strides=list(cfg.DATA.strides), mask=None).astype(np.uint16)

        pred_seg = merge_func(pred_seg)
        pred_seg = relabel(pred_seg.astype(np.uint16))

        seg.append(pred_seg.copy())
        pred_seg = pred_seg.astype(np.uint16)
        gt_ins = gt_ins.astype(np.uint16)
        id_list, id_nums = np.unique(pred_seg,return_counts=True)
        id_list = np.array(id_list)
        id_nums = np.array(id_nums)
        id_nums = id_nums>2000
        remove_id = id_list[id_nums]
        for id in remove_id:
            pred_seg[pred_seg==id] = 0

        pred_seg = pred_seg[92:-92, 4:-4]
        gt_ins = gt_ins[92:-92, 4:-4]
        id = pred_seg[0,0]
        pred_seg[pred_seg==id] = 0

        # evaluate
        if pred_seg.max() == 0:
            temp_aji = 0.0
            temp_dice = 0.0
            temp_f1 = 0.0
            temp_pq = 0.0
        else:
            temp_aji = agg_jc_index(gt_ins, pred_seg)
            temp_f1 = pixel_f1(gt_ins, pred_seg)
            gt_relabel = remap_label(gt_ins, by_size=False)
            pred_relabel = remap_label(pred_seg, by_size=False)
            pq_info_cur = get_fast_pq(gt_relabel, pred_relabel, match_iou=0.5)[0]
            temp_dice = pq_info_cur[0]
            temp_pq = pq_info_cur[2]
            aji_score.append(temp_aji)
            dice_score.append(temp_dice)
            f1_score.append(temp_f1)
            pq_score.append(temp_pq)

        print('image=%d, AJI=%.6f, Dice=%.6f, F1=%.6f, PQ=%.6f' % (k, temp_aji, temp_dice, temp_f1, temp_pq))
        f_txt.write('image=%d, AJI=%.6f, Dice=%.6f, F1=%.6f, PQ=%.6f' % (k, temp_aji, temp_dice, temp_f1, temp_pq))
        f_txt.write('\n')

        val_show_emd(k, batch_data['img1'][0][:,92:-92, 4:-4], emb_f[0][:,92:-92, 4:-4], pred_seg, gt_ins, embs_img_path)
        show_affs_all(k, out_affs[:,92:-92, 4:-4], affs_gt[:,92:-92, 4:-4], affs_img_path)
        show_affs_all(k, entropy_out[:,92:-92, 4:-4], affs_gt[:,92:-92, 4:-4], entropy_img_path)

    aji_score = np.asarray(aji_score)
    dice_score = np.asarray(dice_score)
    f1_score = np.asarray(f1_score)
    pq_score = np.asarray(pq_score)
    mean_aji = np.mean(aji_score)
    std_aji = np.std(aji_score)
    mean_dice = np.mean(dice_score)
    std_dice = np.std(dice_score)
    mean_f1 = np.mean(f1_score)
    std_f1 = np.std(f1_score)
    mean_pq = np.mean(pq_score)
    std_pq = np.std(pq_score)

    cost_time = time.time() - start_time
    epoch_loss = sum(losses_valid) / len(losses_valid)

    print('model-%d, valid-loss=%.6f, AJI=%.6f(%.6f), Dice=%.6f(%.6f), F1=%.6f(%.6f), PQ=%.6f(%.6f)' %  \
        (args.model_id, epoch_loss, mean_aji, std_aji, mean_dice, std_dice, mean_f1, std_f1, mean_pq, std_pq), flush=True)
    print('COST TIME = %.6f' % cost_time)

    f_txt.write('model-%d, valid-loss=%.6f, AJI=%.6f(%.6f), Dice=%.6f(%.6f), F1=%.6f(%.6f), PQ=%.6f(%.6f)' %  \
        (args.model_id, epoch_loss, mean_aji, std_aji, mean_dice, std_dice, mean_f1, std_f1, mean_pq, std_pq))
    f_txt.write('\n')
    f_txt.close()

    seg = np.asarray(seg, dtype=np.uint16)
    f = h5py.File(os.path.join(out_embs, 'seg.hdf'), 'w')
    f.create_dataset('main', data=seg, dtype=seg.dtype, compression='gzip')
    f.close()
