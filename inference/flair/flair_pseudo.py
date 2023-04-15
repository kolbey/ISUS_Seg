'''
熵高的预测位置，代表当前位置的预测结果可信度不高，
因此通过熵值对可信度高的位置进行保留，可信度低的位置填充为忽略的标签类别，
这种伪标签训练方案是不对的，因为熵高的地方忽略标签会误导模型，
更加合理的伪标签训练方案还有待研究！！
'''

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

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


def img_writer(inp):
    (mask,  mask_id) = inp
    mask_png = mask.astype(np.uint8)
    mask_name_png = mask_id
    cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"]) ## lr is flip TTA, d4 is multi-scale TTA

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

            image_ids = input["name_id"]
            raw_predictions = nn.Softmax(dim=1)(raw_predictions)

            entorpy = -torch.sum(raw_predictions * torch.log(raw_predictions + 1e-10), dim=1)
            entorpy_thresh = np.percentile(entorpy.cpu(),10)
            entorpy_mask = entorpy.ge(entorpy_thresh)

            predictions = raw_predictions.argmax(dim=1)
            predictions[entorpy_mask] = config.ignore_index


            for i in range(raw_predictions.shape[0]):
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
