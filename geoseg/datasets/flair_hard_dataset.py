# 多分类语义分割转变为二分类语义分割进行训练
from .transform import *
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import matplotlib.patches as mpatches
from PIL import Image, ImageOps
import random
from pathlib import Path
import json
from random import shuffle
import rasterio

# CLASSES = ('herbaceous vegetation','agricultural land', 'plowed land', 'other')
# PALETTE = [[255, 255, 0], [255, 0, 0], [0, 255, 0], [128, 42, 42]]

CLASSES = ('water', 'other')
PALETTE = [[255, 255, 0], [255, 0, 0]]

ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)


def get_training_transform():
    train_transform = [
        # albu.Resize(height=1024, width=1024),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        albu.RandomRotate90(p=0.5),
        # albu.OneOf([
        #     albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
        #     albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25)
        # ], p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    # # multi-scale training and crop
    # img = Image.fromarray(img)
    # mask = Image.fromarray(mask)
    # crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
    #                     SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    # img, mask = crop_aug(img, mask)

    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        # albu.Resize(height=1024, width=1024, interpolation=cv2.INTER_CUBIC),
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

def val_aug(img, mask):
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_test_transform():
    test_transform = [
        # albu.Resize(height=1024, width=1024, interpolation=cv2.INTER_CUBIC),
        albu.Normalize()
    ]
    return albu.Compose(test_transform)

def test_aug(img):
    aug = get_test_transform()(image=img.copy())
    img = aug['image']
    return img



####### Hard label Train
class FLAIRHardDataset(Dataset):
    def __init__(self, data_root='../flair',
                 path_metadata='../flair/flair-one_metadata.json',
                 mosaic_ratio=0.25, mode = 'train',
                 transform=train_aug, img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.path_metadata = path_metadata
        self.mosaic_ratio = mosaic_ratio
        self.num_classes = len(CLASSES)
        self.mode = mode

        self.transform = transform
        self.img_size = img_size
        self.data_ids = self.load_data(self.data_root, self.path_metadata)

        if self.mode == 'train':
            self.data_ids = self.data_ids[0]
        if self.mode == 'val':
            self.data_ids = self.data_ids[1]
        if self.mode == 'test':
            self.data_ids = self.data_ids[2]

    def __getitem__(self, index):
        p_ratio = random.random()
        img, mask = self.load_img_and_mask(index)
        if p_ratio < self.mosaic_ratio:
            img, mask = self.load_mosaic_img_and_mask(index)
        if self.transform:
            img, mask = self.transform(img, mask)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        results = {'img': img, 'gt_semantic_seg': mask}

        return results

    def __len__(self):
        length = len(self.data_ids["IMG"])
        return length

    def load_data(self, data_root, path_metadata, val_percent=0.9, use_metadata=False):
    
        def _gather_data(path_folders, path_metadata: str, use_metadata: bool, test_set: bool) -> dict:
        
            #### return data paths
            def get_data_paths (path, filter):
                for path in Path(path).rglob(filter):
                    yield path.resolve().as_posix()               
        
            data = {'IMG':[],'MSK':[]}
            for domain in path_folders:
                data['IMG'] += sorted(list(get_data_paths(domain, 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
                if test_set == False:
                    data['MSK'] += sorted(list(get_data_paths(domain, 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
            
            if test_set == False:
                if len(data['IMG']) != len(data['MSK']): 
                    print('[WARNING !!] UNMATCHING NUMBER OF IMAGES AND MASKS ! Please check load_data function for debugging.')
                if data['IMG'][0][-10:-4] != data['MSK'][0][-10:-4] or data['IMG'][-1][-10:-4] != data['MSK'][-1][-10:-4]: 
                    print('[WARNING !!] UNSORTED IMAGES AND MASKS FOUND ! Please check load_data function for debugging.')                
                
            return data
        
        
        path_trainval = Path(data_root, "train")
        trainval_domains = [Path(path_trainval, domain) for domain in os.listdir(path_trainval)]
        shuffle(trainval_domains)
        idx_split = int(len(trainval_domains) * val_percent)
        # train_domains, val_domains = trainval_domains[:idx_split], trainval_domains[idx_split:]
        train_domains, val_domains = trainval_domains[:], trainval_domains[idx_split:]
        
        dict_train = _gather_data(train_domains, path_metadata, use_metadata=use_metadata, test_set=False)
        dict_val = _gather_data(val_domains, path_metadata, use_metadata=use_metadata, test_set=False)
        
        path_test = Path(data_root, "test")
        test_domains = [Path(path_test, domain) for domain in os.listdir(path_test)]
        
        dict_test = _gather_data(test_domains, path_metadata, use_metadata=use_metadata, test_set=True)
        
        return dict_train, dict_val, dict_test

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array

    def read_msk(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_msk:
            # array = src_msk.read()
            array = src_msk.read()[0]
            array[array > 13] = 13
            array = array-1
            array[array < 4] = 1
            array[array > 4] = 1
            array[array == 4] = 0
            # array = array - 9
            # array = np.stack([array == i for i in range(self.num_classes)], axis=0)
            return array

    def load_img_and_mask(self, index):
        img_id = self.data_ids["IMG"][index]
        msk_id = self.data_ids["MSK"][index]
        img = self.read_img(raster_file=img_id)
        img = img[:3, :, :]
        img = torch.from_numpy(img).permute(1, 2, 0).numpy()
        mask = self.read_msk(raster_file=msk_id)
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.data_ids["IMG"]) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        # img = Image.fromarray(img)
        # mask = Image.fromarray(mask)

        return img, mask