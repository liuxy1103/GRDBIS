import os
import cv2
import h5py
import yaml
import torch
import argparse
import numpy as np
import SimpleITK as sitk
from skimage import morphology
from attrdict import AttrDict
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from utils_EM.show import draw_fragments_3d
from provider_valid_EM import Provider_valid
from loss_EM.loss import BCELoss, WeightedBCE, MSELoss, WeightedMSE
# from unet3d_mala import UNet3D_MALA
from utils_EM.shift_channels import shift_func
from data_EM.data_segmentation import seg_widen_border
from loss_EM.loss_embedding_mse import inf_embedding_loss_norm1, inf_embedding_loss_norm5
from data_EM.data_affinity import seg_to_aff
import waterz
from utils_EM.lmc import mc_baseline
from utils_EM.fragment import watershed, randomlabel, relabel
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
from utils_EM.show import *
from data_EM.data_segmentation import seg_widen_border
from utils_EM.cluster import cluster_ms
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

def embedding_pca(embeddings, n_components=3, as_rgb=True):
    if as_rgb and n_components != 3:
        raise ValueError("")

    pca = PCA(n_components=n_components)
    embed_dim = embeddings.shape[0]
    shape = embeddings.shape[1:]

    embed_flat = embeddings.reshape(embed_dim, -1).T
    embed_flat = pca.fit_transform(embed_flat).T
    embed_flat = embed_flat.reshape((n_components,) + shape)

    if as_rgb:
        embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
        embed_flat = embed_flat.astype('uint8')
    return embed_flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_ac34_dics_mala_emb',
                        help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str,
                        default='2023-02-23--04-06-16_seg_ac34_dics_mala_emb')
    parser.add_argument('-id', '--model_id', type=int, default=275000)
    parser.add_argument('-m', '--mode', type=str, default='ac3')  # cremiA,fib2
    parser.add_argument('-ts', '--teacher_student', type=str, default='Student3')  # Teacher,Student1,Student2,Student3
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-ie', '--if_embedding', action='store_true', default=True)
    parser.add_argument('-dl', '--dilate_label', action='store_true', default=False)
    parser.add_argument('-pm', '--pixel_metric', action='store_true', default=False)
    parser.add_argument('-ca', '--crop_affs', action='store_true', default=False)
    parser.add_argument('-sa', '--save_affs', action='store_true', default=False)#
    parser.add_argument('-sg', '--save_gt', action='store_true', default=True)#
    parser.add_argument('-nlmc', '--not_lmc', action='store_true', default=False)
    parser.add_argument('-swa', '--show_aff', action='store_true', default=False)#
    parser.add_argument('-swe', '--show_emb', action='store_true', default=False)#
    parser.add_argument('-sws', '--show_seg', action='store_true', default=True)#
    args = parser.parse_args()
    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)


    if args.teacher_student == 'Teacher':
        from unet3d_mala import UNet3D_MALA_embedding as UNet3D_MALA
    elif args.teacher_student == 'Student1':
        from unet3d_mala import UNet3D_MALA_embedding_small1 as UNet3D_MALA
    elif args.teacher_student == 'Student2':
        from unet3d_mala import UNet3D_MALA_embedding_small2 as UNet3D_MALA
    elif args.teacher_student == 'Student3':
        from unet3d_mala import UNet3D_MALA_embedding_small3 as UNet3D_MALA
    elif args.teacher_student == 'Student4':
        from unet3d_mala import UNet3D_MALA_embedding_small4 as UNet3D_MALA
    elif args.teacher_student == 'Student5':
        from unet3d_mala import UNet3D_MALA_embedding_small5 as UNet3D_MALA
    elif args.teacher_student == 'Student6':
        from unet3d_mala import UNet3D_MALA_embedding_small6 as UNet3D_MALA
    else:
        from unet3d_mala import UNet3D_MALA

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))
    

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    out_path = os.path.join('../inference_extension', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+str(args.model_id)
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
    if cfg.MODEL.model_type == 'mala':
        print('load mala model!')
        if 'embedding' not in args.cfg:
            model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc,
                                if_sigmoid=cfg.MODEL.if_sigmoid,
                                init_mode=cfg.MODEL.init_mode_mala).cuda()
        else:
            model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc,
                                if_sigmoid=cfg.MODEL.if_sigmoid,
                                init_mode=cfg.MODEL.init_mode_mala,
                                emd=cfg.MODEL.emd).cuda()
    else:
        print('load superhuman model!')
        raise NotImplementedError

    ckpt_path = os.path.join('../models_extension', trained_model, 'stu_KD.ckpt')
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        # name = k[7:] # remove module.
        name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.cuda()

    # valid_provider = Provider_valid(cfg, valid_data=args.mode, test_split=args.test_split)
    # val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)

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

    #############################################
    # for MALA
    print('Read Dataset...',args.mode)
    if args.mode=='fib2':
        f = h5py.File('../data/fib2/fib2_inputs.h5', 'r')
        raw_ori = f['main']
        f.close()
        raw = raw_ori
        f = h5py.File('../data/fib2/fib2_labels.h5', 'r')
        gt_seg = f['main'][50:-25]
        f.close()
        raw = np.pad(raw, ((14, 14), (106, 106), (106, 106)), mode='reflect')
    elif args.mode == 'cremiC' or args.mode == 'cremiB' or args.mode == 'cremiA' :
        f = h5py.File('../data/cremi/'+args.mode+'_inputs_interp.h5', 'r')
        raw_ori = f['main'][50:-25]
        f.close()
        raw = raw_ori
        f = h5py.File('../data/cremi/'+args.mode+'_labels.h5', 'r')
        gt_seg = f['main'][50:-25]
        f.close()
        raw = np.pad(raw, ((14,20),(106,106),(106,106)), mode='reflect')

    elif args.mode == 'ac3':
        f = h5py.File('../data/ac3_ac4/'+'AC3_inputs.h5', 'r')
        raw_ori = f['main'][:100]
        f.close()
        raw = raw_ori
        f = h5py.File('../data/ac3_ac4/'+'AC3_labels.h5', 'r')
        gt_seg = f['main'][:100]
        f.close()
        raw = np.pad(raw, ((14,14),(106,106),(106,106)), mode='reflect')

    print('raw shape:', raw.shape)
    stride = 56
    in_shape = [84, 268, 268]
    out_shape = [56, 56, 56]
    output_affs = np.zeros([1, 3, raw.shape[0] - (in_shape[0] - out_shape[0]),
                    raw.shape[1] - (in_shape[1] - out_shape[1]),
                    raw.shape[2] - (in_shape[2] - out_shape[2])], np.float32)
    output_affs = torch.Tensor(output_affs)
    out_embedding = np.zeros([1, 16, raw.shape[0] - (in_shape[0] - out_shape[0]),
                    raw.shape[1] - (in_shape[1] - out_shape[1]),
                    raw.shape[2] - (in_shape[2] - out_shape[2])], np.float32)

    model.eval()
    loss_all = []
    f_txt = open(os.path.join(out_affs, 'scores.txt'), 'w')
    # print('the number of sub-volume:', len(valid_provider))
    # losses_valid = []
    t1 = time.time()
    # pbar = tqdm(total=len(valid_provider))
    time_total = 0
    with torch.no_grad():
        part = 0
        zz = list(np.arange(0, raw.shape[0] - in_shape[0], stride)) + [raw.shape[0] - in_shape[0]]
        for z in zz:
            part += 1
            print('Part %d / %d' % (part, len(zz)))
            for y in tqdm(list(np.arange(0, raw.shape[1] - in_shape[1], stride)) + [raw.shape[1] - in_shape[1]]):
                for x in list(np.arange(0, raw.shape[2] - in_shape[2], stride)) + [raw.shape[2] - in_shape[2]]:
                    input = raw[z : z + in_shape[0], y : y + in_shape[1], x : x + in_shape[2]]
                    input = input.astype(np.float32) / 255.0
                    input = np.expand_dims(np.expand_dims(input, axis=0), axis=0)
                    input = torch.Tensor(input).cuda()
                    if args.if_embedding:
                        time_s = time.time()
                        embedding = model(input)
                        time_e = time.time()
                        time_total += (time_e-time_s)
                        if cfg.TRAIN.embedding_mode == 1:
                            pred = inf_embedding_loss_norm1(embedding)
                        elif cfg.TRAIN.embedding_mode == 5:
                            pred = inf_embedding_loss_norm5(embedding)
                        else:
                            raise NotImplementedError
                        shift = 1
                        pred[:, 1, :, :shift, :] = pred[:, 1, :, shift:shift*2, :]
                        pred[:, 2, :, :, :shift] = pred[:, 2, :, :, shift:shift*2]
                        pred[:, 0, :shift, :, :] = pred[:, 0, shift:shift*2, :, :]
                        pred = F.relu(pred)
                    else:
                        pred = model(input)

                    output_affs[:, :, z : z + out_shape[0], y : y + out_shape[1], x : x + out_shape[2]] = pred.data.cpu()
                    out_embedding[:, :, z: z + out_shape[0], y: y + out_shape[1], x: x + out_shape[2]] = embedding.data.cpu()
    output_affs = torch.nn.functional.pad(output_affs, ((in_shape[2] - out_shape[2]) // 2, (in_shape[2] - out_shape[2]) // 2,
                                            (in_shape[1] - out_shape[1]) // 2, (in_shape[1] - out_shape[1]) // 2,
                                            (in_shape[0] - out_shape[0]) // 2, (in_shape[0] - out_shape[0]) // 2))
    
    # format
    output_affs = output_affs.data.numpy()
    output_affs = np.squeeze(output_affs, axis=0)
    output_affs[output_affs>1] = 1
    output_affs[output_affs<0] = 0


    cost_time = time.time() - t1
    print('Inference time=%.6f' % cost_time)
    f_txt.write('Inference time=%.6f' % cost_time)
    f_txt.write('\n')
    print('Model Inference time=%.6f' % time_total)
    f_txt.write('Model Inference time=%.6f' % time_total)
    f_txt.write('\n')
    epoch_loss = 0.0
    lb = gt_seg.copy()
    lb = seg_widen_border(lb, tsz_h=1)
    if cfg.MODEL.output_nc == 3:
        lb_affs = seg_to_aff(lb).astype(np.float32)
    elif cfg.MODEL.output_nc == 12:
        nhood233 = np.asarray([-2, 0, 0, 0, -3, 0, 0, 0, -3]).reshape((3, 3))
        nhood399 = np.asarray([-3, 0, 0, 0, -9, 0, 0, 0, -9]).reshape((3, 3))
        nhood427 = np.asarray([-4, 0, 0, 0, -27, 0, 0, 0, -27]).reshape((3, 3))
        label111 = seg_to_aff(lb, pad='').astype(np.float32)
        label233 = seg_to_aff(lb, nhood233, pad='')
        label399 = seg_to_aff(lb, nhood399, pad='')
        label427 = seg_to_aff(lb, nhood427, pad='')
        lb_affs = np.concatenate((label111, label233, label399, label427), axis=0)
    gt_affs = lb_affs

    if args.dilate_label:
        print('dilate labels')
        gt_seg = seg_widen_border(gt_seg, tsz_h=1)
    # gt_seg = gt_seg[14:-14,106:-106,106:-106]

    # for mutex
    output_affs = output_affs[:3]
    output_affs = output_affs[:, 14:-14,106:-106,106:-106]
    # if args.crop_affs:
    #     output_affs = output_affs[:, 14:-14,106:-106,106:-106]
    # else:
    #     for d in range(3):
    #         output_affs[d][mask] = 0
    if args.mode == 'cremiC' or args.mode == 'cremiB' or args.mode == 'cremiA':
        output_affs = output_affs[:,:-6]
    print('affinity shape:', output_affs.shape)

    # save
    if args.save_affs:
        print('save affs...')
        # print('the shape of affs:', output_affs.shape)
        f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
        f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
        f.close()

    print('segmentation...')
    fragments = watershed(output_affs, 'maxima_distance')

 
    sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
    segmentation = list(waterz.agglomerate(output_affs, [args.threshold],
                                        fragments=fragments,
                                        scoring_function=sf,
                                        discretize_queue=256))[0]
    segmentation = relabel(segmentation).astype(np.uint64)
    print('the max id = %d' % np.max(segmentation))
    f = h5py.File(os.path.join(out_affs, 'seg_waterz.hdf'), 'w')
    f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
    f.close()
    if args.save_gt:
        f = h5py.File(os.path.join(out_affs, 'seg_gt.hdf'), 'w')
        f.create_dataset('main', data=gt_seg, dtype=gt_seg.dtype, compression='gzip')
        f.close()
    # if not args.crop_affs:
    #     segmentation = segmentation[14:-14,106:-106,106:-106]
    arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    voi_sum = voi_split + voi_merge
    print('waterz: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
        (voi_split, voi_merge, voi_sum, arand))
    f_txt.write('waterz: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
        (voi_split, voi_merge, voi_sum, arand))
    f_txt.write('\n')

    if not args.not_lmc:
        segmentation = mc_baseline(output_affs)
        segmentation = relabel(segmentation).astype(np.uint64)
        print('the max id = %d' % np.max(segmentation))
        f = h5py.File(os.path.join(out_affs, 'seg_lmc.hdf'), 'w')
        f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
        f.close()

        # if not args.crop_affs:
        #     segmentation = segmentation[14:-14,106:-106,106:-106]
        arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
        voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
        voi_sum = voi_split + voi_merge
        print('LMC: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (voi_split, voi_merge, voi_sum, arand))
        f_txt.write('LMC: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (voi_split, voi_merge, voi_sum, arand))
        f_txt.write('\n')

    # compute MSE
    if args.pixel_metric:
        print('MSE...')
        output_affs_prop = output_affs.copy()
        whole_mse = np.sum(np.square(output_affs - gt_affs)) / np.size(gt_affs)
        print('BCE...')
        output_affs = np.clip(output_affs, 0.000001, 0.999999)
        bce = -(gt_affs * np.log(output_affs) + (1 - gt_affs) * np.log(1 - output_affs))
        whole_bce = np.sum(bce) / np.size(gt_affs)
        output_affs[output_affs <= 0.5] = 0
        output_affs[output_affs > 0.5] = 1
        print('F1...')
        whole_arand = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), output_affs.astype(np.uint8).flatten())
        # whole_arand = 0.0
        # new
        print('F1 boundary...')
        whole_arand_bound = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs.astype(np.uint8).flatten())
        # whole_arand_bound = 0.0
        print('mAP...')
        whole_map = average_precision_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
        # whole_map = 0.0
        print('AUC...')
        whole_auc = roc_auc_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
        # whole_auc = 0.0
        ###################################################
        malis = 0.0
        ###################################################
        print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
            (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
        f_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
                    (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
        f_txt.write('\n')
    else:
        output_affs_prop = output_affs
    f_txt.close()

            
    print('Done')
