import sys
sys.path.append("/data_raid5_21T/zhangzewen/lu/GeoLab/")
import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.FTUNetFormer import ft_unetformer
from geoseg.models.DeepLabv3_plus import DeepLabv3_plus
from geoseg.models.DCSwin import dcswin_small


def img_writer(inp):
    (mask,  mask_id) = inp
    mask_png = mask.astype(np.uint8)
    mask_name_png = mask_id + '.png'
    cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default='d4', choices=[None, "d4", "lr"]) ## lr is flip TTA, d4 is multi-scale TTA
    arg("--rgb", help="whether output rgb masks", action='store_true')
    arg("--val", help="whether eval validation set", action='store_true')
    return parser.parse_args()

def model_load(train_dict, state_dict):
    model_dict = {}
    for k, v in train_dict.items():
        new_k = k[4:]
        if new_k in state_dict:
            model_dict[new_k] = v
    state_dict.update(model_dict)
    return model_dict

def load_checkpoint(checkpoint_path, model):
    pretrained_dict = torch.load(checkpoint_path)['model_state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    model_unet = UNetFormer(num_classes=7)
    checkpoint_unet = torch.load('model_weights/loveda/UN-V0/UN-V0.ckpt')
    model_dict_unet = model_load(checkpoint_unet['state_dict'], model_unet.state_dict())
    model_unet.load_state_dict(model_dict_unet)
    model_unet.cuda(config.gpus[0])
    model_unet.eval()

    model_ftunet = ft_unetformer(num_classes=7, decoder_channels=256)
    checkpoint_ftunet = torch.load('model_weights/loveda/FT-V0/FT-V0.ckpt', map_location={'cuda:2':'cuda:3'})
    model_dict_ftunet = model_load(checkpoint_ftunet['state_dict'], model_ftunet.state_dict())
    model_ftunet.load_state_dict(model_dict_ftunet)
    model_ftunet.cuda(config.gpus[0])
    model_ftunet.eval()

    model_dl = DeepLabv3_plus(nInputChannels=3, n_classes=7, os=16, pretrained=True, _print=True)
    checkpoint_dl = torch.load('model_weights/loveda/DL-V0/DL-V0.ckpt', map_location={'cuda:1':'cuda:3'})
    model_dict_dl = model_load(checkpoint_dl['state_dict'], model_dl.state_dict())
    model_dl.load_state_dict(model_dict_dl)
    model_dl.cuda(config.gpus[0])
    model_dl.eval()

    model_dc = dcswin_small(num_classes=7)
    checkpoint_dc = torch.load('model_weights/loveda/DC-V0/DC-V0.ckpt', map_location={'cuda:0':'cuda:3'})
    model_dict_dc = model_load(checkpoint_dc['state_dict'], model_dc.state_dict())
    model_dc.load_state_dict(model_dict_dc)
    model_dc.cuda(config.gpus[0])
    model_dc.eval()
    
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
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
                # tta.Scale(scales=[0.75, 1.0, 1.25, 1.5, 1.75], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model_unet = tta.SegmentationTTAWrapper(model_unet, transforms)
        model_ftunet = tta.SegmentationTTAWrapper(model_ftunet, transforms)
        model_dl = tta.SegmentationTTAWrapper(model_dl, transforms)
        model_dc = tta.SegmentationTTAWrapper(model_dc, transforms)

    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions_unet = model_unet(input['img'].cuda(config.gpus[0]))
            raw_predictions_ftunet = model_ftunet(input['img'].cuda(config.gpus[0]))
            raw_predictions_dl = model_dl(input['img'].cuda(config.gpus[0]))
            raw_predictions_dc = model_dc(input['img'].cuda(config.gpus[0]))

            image_ids = input["img_id"]

            raw_predictions = (raw_predictions_dl + raw_predictions_dc + raw_predictions_unet + raw_predictions_ftunet) / 4.0

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path / mask_name)))

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()