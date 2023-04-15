import sys
sys.path.append("/data_raid5_21T/zhangzewen/lu/GeoLab/")
from tools.cfg import py2cfg
import argparse
from pathlib import Path
import glob
from PIL import Image
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu
from catalyst.dl import SupervisedRunner
from skimage.morphology import remove_small_holes, remove_small_objects
# from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision import *
import random
import os


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image_path", type=Path, required=True, help="Path to  huge image folder")
    arg("-m", "--mask_path", type=Path, required=True, help="Path to  huge mask folder")
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("-ph", "--patch-height", help="height of patch size", type=int, default=512)
    arg("-pw", "--patch-width", help="width of patch size", type=int, default=512)
    arg("-b", "--batch-size", help="batch size", type=int, default=2)
    return parser.parse_args()


def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    # print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                           border_mode=0, value=[0, 0, 0])(image=image)
    img_pad = pad['image']
    return img_pad, height_pad, width_pad


class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, transform=albu.Normalize()):
        self.tile_list = tile_list
        self.transform = transform

    def __getitem__(self, index):
        img = self.tile_list[index]
        img_id = index
        aug = self.transform(image=img)
        img = aug['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img)
        return results

    def __len__(self):
        return len(self.tile_list)


def make_dataset_for_one_huge_image(img_path, patch_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tile_list = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)

    output_height, output_width = image_pad.shape[0], image_pad.shape[1]

    for x in range(0, output_height, patch_size[0]):
        for y in range(0, output_width, patch_size[1]):
            image_tile = image_pad[x:x+patch_size[0], y:y+patch_size[1]]
            tile_list.append(image_tile)

    dataset = InferenceDataset(tile_list=tile_list)
    return dataset, width_pad, height_pad, output_width, output_height, image_pad, img.shape


def main():
    args = get_args()
    seed_everything(42)
    patch_size = (args.patch_height, args.patch_width)
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)

    model.cuda(config.gpus[0])
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1, 1.25, 1.5, 1.75]),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    img_paths = []
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    for ext in ('*.tif', '*.png', '*.jpg'):
        img_paths.extend(glob.glob(os.path.join(args.image_path, ext)))
    img_paths.sort()

    # print(img_paths)
    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        mask_name = img_name.split('.')[0] + '.png'
        mask_path = os.path.join(args.mask_path, mask_name)
        # print('origin mask', original_mask.shape)
        dataset, width_pad, height_pad, output_width, output_height, img_pad, img_shape = \
            make_dataset_for_one_huge_image(img_path, patch_size)
        # print('img_padded', img_pad.shape)
        output_mask = np.zeros(shape=(output_height, output_width), dtype=np.uint8)
        output_tiles = []
        k = 0
        with torch.no_grad():
            dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                    drop_last=False, shuffle=False)
            for input in tqdm(dataloader):
                # raw_prediction NxCxHxW
                raw_predictions = model(input['img'].cuda(config.gpus[0]))
                # print('raw_pred shape:', raw_predictions.shape)
                raw_predictions = nn.Softmax(dim=1)(raw_predictions)
                # input_images['features'] NxCxHxW C=3
                predictions = raw_predictions.argmax(dim=1)
                predictions = predictions * 255
                image_ids = input['img_id']
                # print('prediction', predictions.shape)
                # print(np.unique(predictions))

                for i in range(predictions.shape[0]):
                    mask = predictions[i].cpu().numpy()
                    output_tiles.append((mask, image_ids[i].cpu().numpy()))

        for m in range(0, output_height, patch_size[0]):
            for n in range(0, output_width, patch_size[1]):
                output_mask[m:m + patch_size[0], n:n + patch_size[1]] = output_tiles[k][0]
                # print(output_tiles[k][1])
                k = k + 1

        output_mask = output_mask[-img_shape[0]:, -img_shape[1]:]

        # if height_pad != 0 and width_pad == 0:
        #     h_index = height_pad // 2
        #     output_mask = output_mask[h_index:-h_index, :]
        # elif height_pad == 0 and width_pad != 0:
        #     w_index = width_pad // 2
        #     output_mask = output_mask[:, w_index:-w_index]
        # elif height_pad != 0 and width_pad != 0:
        #     h_index = height_pad // 2
        #     w_index = width_pad // 2
        #     output_mask = output_mask[h_index:-h_index:, w_index:-w_index]
        # else:
        #     output_mask = output_mask

        # print('mask', output_mask.shape)

        output_mask = output_mask


        output_label = Image.open(mask_path).convert('L')
        output_label = np.array(output_label)
        output_map = np.zeros(shape=(output_height, 2 * output_width), dtype=np.uint8)
        output_map[0:output_height, 0:output_width] = output_label
        output_map[0:output_height, output_width:2*output_width] = output_mask
        cv2.imwrite(os.path.join(args.output_path, img_name), output_map)


if __name__ == "__main__":
    main()
