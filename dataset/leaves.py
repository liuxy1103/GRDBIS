import os
import numpy as np
from PIL import Image
from dataset.transforms import RandomAffine
from dataset.dataset import MyDataset
import glob


class LeavesDataset(MyDataset):

    def __init__(self,
                 leaves_dir='',
                 leaves_test_dir='',
                 batch_size=1,
                 gt_maxseqlen=20,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split='train',
                 resize=False,
                 imsize=256,
                 rotation=10,
                 translation=0.1,
                 shear=0.1,
                 zoom=0.7,
                 num_train=108,
                 padding=True):

        CLASSES = ['<eos>', 'leaf']

        self.split = split
        self.num_train = num_train
        self.padding = padding
        self.classes = CLASSES
        self.num_classes = len(self.classes)
        self.max_seq_len = gt_maxseqlen
        # self.image_dir = os.path.join(leaves_dir, 'A1')
        self.image_dir = leaves_dir
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.no_run_coco_eval = True
        if self.batch_size == 1:
            self.crop = False
        else:
            self.crop = True
        self.flip = augment
        if augment and not resize:
            self.augmentation_transform = RandomAffine(rotation_range=rotation,
                                                    translation_range=translation,
                                                    shear_range=shear,
                                                    interp='nearest')
        elif augment and resize:
            self.augmentation_transform = RandomAffine(rotation_range=rotation,
                                                    translation_range=translation,
                                                    shear_range=shear,
                                                    zoom_range=(zoom,1),
                                                    interp='nearest')

        else:
            self.augmentation_transform = None

        self.zoom = zoom
        self.augment = augment
        self.imsize = imsize
        self.resize = resize

        self.image_files = []
        self.gt_files = []
        self.train_image_files = []
        self.train_gt_files = []
        self.val_image_files = []
        self.val_gt_files = []
        self.test_image_files = []
        self.test_mask_files = []

        self.image_total_files = glob.glob(os.path.join(leaves_dir, '*_rgb.png'))
        self.gt_total_files = [w.replace('_rgb', '_label') for w in self.image_total_files]

        self.test_image_total_files = glob.glob(os.path.join(leaves_test_dir, '*_rgb.png'))
        self.test_mask_total_files = [w.replace('_rgb', '_fg') for w in self.test_image_total_files]

        """
        we split the training set between validation and training. The first 96 images will be for training and
        the other will be for validation
        """
        for i in range(len(self.image_total_files)):
            if i < self.num_train:
                self.train_image_files.append(self.image_total_files[i])
                self.train_gt_files.append(self.gt_total_files[i])
            else:
                self.val_image_files.append(self.image_total_files[i])
                self.val_gt_files.append(self.gt_total_files[i])

        for j in range(len(self.test_image_total_files)):
            self.test_image_files.append(self.test_image_total_files[j])
            self.test_mask_files.append(self.test_mask_total_files[j])

        if split == "train":
            self.image_files = self.train_image_files
            self.gt_files = self.train_gt_files
        elif split == "val":
            self.image_files = self.val_image_files
            self.gt_files = self.val_gt_files
        elif split == "test":
            self.image_files = self.test_image_files
            self.gt_files = self.test_mask_files

    def get_raw_sample(self,index):
        """
        Returns sample data in raw format (no resize)
        """
        # image_file = os.path.join(self.image_dir, self.image_files[index])
        image_file = self.image_files[index]
        img = Image.open(image_file).convert('RGB')

        gt_file = self.gt_files[index]
        gt = np.array(Image.open(gt_file))

        if self.padding:
            img = np.asarray(img)
            img = np.pad(img, ((7,7),(22,22),(0,0)), mode='reflect')
            gt = np.pad(gt, ((7,7),(22,22)), mode='constant')

            # if self.split == 'train':
            #     img = np.pad(img, ((200,200),(200,200),(0,0)), mode='reflect')
            img = Image.fromarray(img)

        seg = gt.copy()
        if self.split != "test":
            ins = gt.copy()
            seg[seg > 0] = 1
            return img, ins, seg
        else:
            return img, seg, seg
