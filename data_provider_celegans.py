import os
import sys
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tifffile
from skimage import io
# from data.augmentation import Flip
# from data.augmentation import Elastic
# from data.augmentation import Grayscale
# from data.augmentation import Rotate
# from data.augmentation import Rescale
from utils.affinity_ours import multi_offset, gen_affs_ours
from data.data_segmentation import seg_widen_border, weight_binary_ratio
from utils.utils import remove_list
from utils.neighbor import get_neighbor_by_distance
from data.data_segmentation import relabel
from data.spoco_transform import RgbToLabel, Relabel, GaussianBlur, ImgNormalize, LabelToTensor
from data.augmentation import Flip
from data.augmentation import Elastic
from data.augmentation import Grayscale
from data.augmentation import Rotate
from data.augmentation import Rescale
from data.utils.utils import center_crop_2d
from data.utils.affinity_ours import multi_offset, gen_affs_ours
from data.data.data_segmentation import seg_widen_border, weight_binary_ratio
from data.data.data_consistency import Filp_EMA
from data.utils.utils import remove_list
from data.utils.consistency_aug import tensor2img, img2tensor, add_gauss_noise
from data.utils.consistency_aug import add_gauss_blur, add_intensity, add_mask



def cvppp_sample_instances(label, instance_ratio, random_state, ignore_labels=(0,)):
    # # convert PIL image to np.array
    # label_img = np.array(pil_img)

    # # convert RGB to int
    # label = RgbToLabel()(label_img)
    # relabel
    label = Relabel(run_cc=False)(label)

    unique = np.unique(label)
    for il in ignore_labels:
        unique = np.setdiff1d(unique, il)

    # shuffle labels
    random_state.shuffle(unique)
    # pick instance_ratio objects
    num_objects = round(instance_ratio * len(unique))   #ratio of sampling
    if num_objects == 0:
        # if there are no objects left, just return an empty patch
        return np.zeros_like(label)

    # sample the labels
    sampled_instances = unique[:num_objects]

    mask = np.zeros_like(label)
    # keep only the sampled_instances
    for si in sampled_instances:
        mask[label == si] = 1
    # mask each channel
    mask = mask.astype('uint8')
    # mask = np.stack([mask] * 3, axis=2)
    label_img = label * mask

    return label_img.astype(np.uint8)


class DimExtender:
    def __init__(self, should_extend):
        self.should_extend = should_extend

    def __call__(self, m):
        if self.should_extend:
            return m.unsqueeze(0)
        return m


class Train(Dataset):
    def __init__(self, cfg, mode='train'):
        super(Train, self).__init__()
        self.size = cfg.DATA.size
        self.data_folder = cfg.DATA.data_folder
        self.mode = mode
        self.padding = cfg.DATA.padding
        self.num_train = cfg.DATA.num_train
        self.separate_weight = cfg.DATA.separate_weight
        self.offsets = multi_offset(list(cfg.DATA.shifts), neighbor=cfg.DATA.neighbor)
        self.instance_ratio = cfg.DATA.instance_ratio
        # if (self.mode != "train") and (self.mode != "validation") and (self.mode != "test"):
        #     raise ValueError("The value of dataset mode must be assigned to 'train' or 'validation'")
        if (self.mode != "train") and (self.mode != "validation") and (self.mode != "test"):
            raise ValueError("The value of dataset mode must be assigned to 'train' or 'validation'")

        self.dir_img = os.path.join(self.data_folder, 'images')
        # self.dir_lb = os.path.join(self.data_folder, 'masks')
        self.dir_lb = os.path.join(self.data_folder, 'masks')
        self.dir_meta = os.path.join(self.data_folder, 'metadata')


        # augmentation
        self.if_scale_aug = cfg.DATA.if_scale_aug
        self.if_filp_aug = cfg.DATA.if_filp_aug
        self.if_elastic_aug = cfg.DATA.if_elastic_aug
        self.if_intensity_aug = cfg.DATA.if_intensity_aug
        self.if_rotation_aug = cfg.DATA.if_rotation_aug
        self.augs_init()
        if self.mode == "train":
            f_txt = open(os.path.join(self.dir_meta, 'training.txt'), 'r')
            self.id_img = [x[:-5] for x in f_txt.readlines()]  # remove .png and \n
            f_txt.close()
        elif self.mode == "validation":
            # f_txt = open(os.path.join(self.dir_meta, 'validation.txt'), 'r')
            # valid_set = [x[:-5] for x in f_txt.readlines()]  # remove .png and \n
            # f_txt.close()

            # use test set as valid set directly
            f_txt = open(os.path.join(self.dir_meta, 'validation.txt'), 'r')
            self.id_img = [x[:-5] for x in f_txt.readlines()]  # remove .png and \n
            f_txt.close()
        elif self.mode == "test":
            f_txt = open(os.path.join(self.dir_meta, 'validation.txt'), 'r')
            self.id_img = [x[:-5] for x in f_txt.readlines()]  # remove .png and \n
            f_txt.close()
        else:
            raise NotImplementedError
        print('The number of %s image is %d' % (self.mode, len(self.id_img)))
        self.ema_flip = Filp_EMA()

        # padding for random rotation
        self.crop_size = [cfg.DATA.size, cfg.DATA.size]
        self.crop_from_origin = [0, 0]
        self.padding = cfg.DATA.padding
        self.crop_from_origin[0] = self.crop_size[0] + 2 * self.padding
        self.crop_from_origin[1] = self.crop_size[1] + 2 * self.padding
        self.img_size = [520+2*self.padding, 696+2*self.padding]
   
        # augmentation initoalization
        

        self.label_list = [Image.open(os.path.join(self.dir_lb, self.id_img[k]+'.tif.tif')) for k in range(len(self.id_img))]


        if self.mode == "train":
            if self.instance_ratio is not None:
                assert 0 < self.instance_ratio <= 1
                rs = np.random.RandomState(cfg.TRAIN.manual_seed)
                self.label_list = [cvppp_sample_instances(m, self.instance_ratio, rs) for m in  self.label_list]
            # self.train_label_transform = transforms.Compose(
        print('The number of %s image is %d' % (self.mode, len(self.label_list)))

    def augs_mix(self, data):
        if random.random() > 0.5:
            data = self.aug_flip(data)
        if random.random() > 0.5:
            data = self.aug_rotation(data)
        # if random.random() > 0.5:
        #     data = self.aug_rescale(data)
        if random.random() > 0.5:
            data = self.aug_elastic(data)
        if random.random() > 0.5:
            data = self.aug_grayscale(data)
        return data

    def augs_init(self,):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5)
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
        self.aug_grayscale = Grayscale(p=0.75)


    def __getitem__(self, idx):
        k = random.randint(0, len(self.id_img)-1)
        # read raw image
        imgs = io.imread(os.path.join(self.dir_img, self.id_img[k]+'.png'))
        # label = Image.open(os.path.join(self.dir, self.id_label[k]))
        imgs = imgs.astype(np.float32)
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
        # label = np.asarray(Image.open(os.path.join(self.dir_lb, self.id_img[k]+'.png')))
        label = self.label_list[k]

        # raw images padding
        imgs = np.pad(imgs, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
        label = np.pad(label, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')

        random_x = random.randint(0, self.img_size[0]-self.crop_from_origin[0])
        random_y = random.randint(0, self.img_size[1]-self.crop_from_origin[1])
        imgs = imgs[random_x:random_x+self.crop_from_origin[0], \
                    random_y:random_y+self.crop_from_origin[1]]
        label = label[random_x:random_x+self.crop_from_origin[0], \
                    random_y:random_y+self.crop_from_origin[1]]

        data = {'image': imgs, 'label': label}
        if np.random.rand() < 0.8:
            data = self.augs_mix(data)
        imgs = data['image']
        label = data['label']
        imgs = center_crop_2d(imgs, det_shape=self.crop_size)
        label = center_crop_2d(label, det_shape=self.crop_size)
        imgs = imgs[np.newaxis, :, :]
        img1 = np.repeat(imgs, 3, 0)
        img2 = img1.copy()

        label_numpy = label.copy()
        lb_affs, affs_mask = gen_affs_ours(label_numpy, offsets=self.offsets, ignore=False, padding=True)
        if self.separate_weight:
            weightmap = np.zeros_like(lb_affs)
            for i in range(len(self.offsets)):
                weightmap[i] = weight_binary_ratio(lb_affs[i])   # weight for the foreground class
        else:
            weightmap = weight_binary_ratio(lb_affs)

        if random.random()>0.7:
            img2 = add_gauss_noise(img2)

        if random.random()>0.7:
            img2 = add_gauss_blur(img2)

        if random.random()>0.7:
            img2 = add_intensity(img2)

        if random.random()>0.7:
            label_mask = label_numpy.copy()
            label_mask[label_mask != 0] = 1
            img2 = add_mask(img2, label_mask)


        # label_numpy = np.squeeze(label.numpy())
        label_numpy = relabel(label_numpy)
        label = label_numpy[np.newaxis, ...]
        label = torch.from_numpy(label.astype(np.float32))
        img1 = torch.from_numpy(np.ascontiguousarray(img1, dtype=np.float32))
        img2 = torch.from_numpy(np.ascontiguousarray(img2, dtype=np.float32))
        lb_affs = torch.from_numpy(lb_affs)
        weightmap = torch.from_numpy(weightmap)
        affs_mask = torch.from_numpy(affs_mask)
        return {'image1': img1,
                'image2': img2,
                'seg': label,
                'affs': lb_affs,
                'wmap': weightmap,
                'seg': label,
                'seg_ori': img1,
                'mask': affs_mask
                }

    def __len__(self):
        # return len(self.id_img)
        return int(sys.maxsize)


class Validation(Train):
    def __init__(self, cfg, mode='validation'):
        super(Validation, self).__init__(cfg, mode)
        self.mode = mode

    def __getitem__(self, k):

        imgs = io.imread(os.path.join(self.dir_img, self.id_img[k]+'.png'))
        # normalize to [0, 1]
        imgs = imgs.astype(np.float32)
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
        # read label (the label is converted to instances)
        label = np.asarray(Image.open(os.path.join(self.dir_lb, self.id_img[k]+'.tif.tif')))

        imgs = np.pad(imgs, ((92,92),(4,4)), mode='constant')  
        label = np.pad(label, ((92,92),(4,4)), mode='constant')
        label_numpy = np.squeeze(label)
        label_numpy = relabel(label_numpy)
        lb_affs, affs_mask = gen_affs_ours(label_numpy, offsets=self.offsets, ignore=False, padding=True)
        if self.separate_weight:
            weightmap = np.zeros_like(lb_affs)
            for i in range(len(self.offsets)):
                weightmap[i] = weight_binary_ratio(lb_affs[i])
        else:
            weightmap = weight_binary_ratio(lb_affs)
        imgs = imgs[np.newaxis, :, :]
        imgs = np.repeat(imgs, 3, 0)
        # img2 = add_intensity(imgs)
        img1 = torch.from_numpy(imgs)
        img2 = torch.from_numpy(imgs)

        label = torch.from_numpy(label[np.newaxis, :, :].astype(np.float32))
        lb_affs = torch.from_numpy(lb_affs)
        weightmap = torch.from_numpy(weightmap)
        affs_mask = torch.from_numpy(affs_mask)
        return {'img1': img1,
                'img2': img2,
                'affs': lb_affs,
                'wmap': weightmap,
                'seg': label,
                'mask': affs_mask
                }
       
    def __len__(self):
        return len(self.id_img)

def collate_fn(batchs):

    batch_img1 = []
    batch_img2 = []
    batch_affs = []
    batch_wmap = []
    batch_seg = []
    batch_seg_ori = []
    batch_mask = []

    for batch in batchs:
        batch_img1.append(batch['image1'])
        batch_img2.append(batch['image2'])
        batch_affs.append(batch['affs'])
        batch_wmap.append(batch['wmap'])
        batch_seg.append(batch['seg'])
        batch_seg_ori.append(batch['seg_ori'])
        batch_mask.append(batch['mask'])


    batch_img1 = torch.stack(batch_img1, 0)
    batch_img2 = torch.stack(batch_img2, 0)
    batch_affs = torch.stack(batch_affs, 0)
    batch_wmap = torch.stack(batch_wmap, 0)
    batch_seg = torch.stack(batch_seg, 0)
    batch_seg_ori = torch.stack(batch_seg_ori, 0)
    batch_mask = torch.stack(batch_mask, 0)


    return {'img1': batch_img1,
            'img2': batch_img2,
            'affs': batch_affs,
            'wmap': batch_wmap,
            'seg': batch_seg,
            'seg_ori': batch_seg_ori,
            'mask': batch_mask}

class Provider(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        if self.stage == 'train':
            self.data = Train(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'valid':
            pass
        else:
            raise AttributeError('Stage must be train/valid')
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        # return self.data.num_per_epoch
        return int(sys.maxsize)

    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                            shuffle=False, collate_fn=collate_fn, drop_last=False, pin_memory=True))
        else:
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                            shuffle=False, collate_fn=collate_fn, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            return batch


def show_batch(temp_data, out_path):
    tmp_data = temp_data['image']
    affs = temp_data['affs']
    weightmap = temp_data['wmap']
    # seg = temp_data['mask']
    seg = temp_data['seg']

    tmp_data = tmp_data.numpy()
    tmp_data = show_raw_img(tmp_data)

    shift = -1
    seg = np.squeeze(seg.numpy().astype(np.uint8))
    # seg = seg[shift]
    # seg = seg[:,:,np.newaxis]
    # seg = np.repeat(seg, 3, 2)
    # seg_color = (seg * 255).astype(np.uint8)
    seg_color = draw_fragments_2d(seg)

    affs = np.squeeze(affs.numpy())
    affs = affs[shift]
    affs = affs[:,:,np.newaxis]
    affs = np.repeat(affs, 3, 2)
    affs = (affs * 255).astype(np.uint8)

    im_cat = np.concatenate([tmp_data, seg_color, affs], axis=1)
    Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))

if __name__ == "__main__":
    import yaml
    from attrdict import AttrDict
    from utils.show import show_raw_img, draw_fragments_2d
    seed = 555
    np.random.seed(seed)
    random.seed(seed)

    cfg_file = 'cvppp_embedding_mse_ours_wmse_mw0_l201.yaml'
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))
    
    out_path = os.path.join('./', 'data_temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    data = Train(cfg)
    for i in range(0, 50):
        temp_data = iter(data).__next__()
        show_batch(temp_data, out_path)

    # data = Validation(cfg, mode='validation')
    # for i, temp_data in enumerate(data):
    #     show_batch(temp_data, out_path)
