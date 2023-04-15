import sys
sys.path.append("/home/liululu/medlab/DATASET/one/GeoLab")
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
import scipy.io as io

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def img_writer(inp):
    (mask,  mask_id) = inp
    mask_png = mask.astype(np.uint8)
    mask_name_png = mask_id
    cv2.imwrite(mask_name_png, mask_png)

def mat_writer(inp):
    (mask,  mask_id) = inp
    mask_png = mask.astype(np.uint8)
    mask_name_png = mask_id.split('.')[0] + '.mat'
    io.savemat(mask_name_png, {'data': mask_png})


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"]) ## lr is flip TTA, d4 is multi-scale TTA
    arg("--rgb", help="whether output rgb masks", action='store_true')
    arg("--val", help="whether eval validation set", action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

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
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
                # tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset
    if args.val:
        evaluator = Evaluator(num_class=config.num_classes)
        evaluator.reset()
        test_dataset = config.val_dataset

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
            raw_predictions = model(input['img'].cuda(config.gpus[0]))

            image_ids = input["img_id"]

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                mask_name = image_ids[i]
                
                results.append((mask, str(args.output_path / mask_name)))

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(mat_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()
