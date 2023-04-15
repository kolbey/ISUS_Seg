'''
多模型推理加权融合，
实际效果很不错，
比赛必备方案！
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

from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.FTUNetFormer import ft_unetformer
from geoseg.models.DeepLabv3_plus import DeepLabv3_plus
from geoseg.models.DCSwin import dcswin_small
from geoseg.models.Unet_plusplus import UnetPlusPlus


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
    pretrained_dict = torch.load(checkpoint_path)['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[4:]: v for k, v in pretrained_dict.items() if k[4:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    model_ftunet_1 = ft_unetformer(num_classes=13, decoder_channels=256)
    checkpoint_ftunet_1 = torch.load('model_weights/flair2/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_1-Fake10/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_1-Fake10.ckpt', map_location={'cuda:0':'cuda:7'})
    model_dict_ftunet_1 = model_load(checkpoint_ftunet_1['state_dict'], model_ftunet_1.state_dict())
    model_ftunet_1.load_state_dict(model_dict_ftunet_1)
    # model_ftunet_1 = load_checkpoint('model_weights/flair2/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_1-Fake10/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_1-Fake10.ckpt', model_ftunet_1)
    model_ftunet_1.cuda(config.gpus[0])
    model_ftunet_1.eval()

    model_ftunet_2 = ft_unetformer(num_classes=13, decoder_channels=256)
    checkpoint_ftunet_2 = torch.load('model_weights/flair2/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_2-Fake10/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_2-Fake10.ckpt', map_location={'cuda:2':'cuda:7'})
    model_dict_ftunet_2 = model_load(checkpoint_ftunet_2['state_dict'], model_ftunet_2.state_dict())
    model_ftunet_2.load_state_dict(model_dict_ftunet_2)
    # model_ftunet_2 = load_checkpoint('model_weights/flair2/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_2/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_2.ckpt', model_ftunet_2)
    model_ftunet_2.cuda(config.gpus[0])
    model_ftunet_2.eval()

    model_ftunet_3 = ft_unetformer(num_classes=13, decoder_channels=256)
    checkpoint_ftunet_3 = torch.load('model_weights/flair2/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_3-Fake10/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_3-Fake10.ckpt', map_location={'cuda:3':'cuda:7'})
    model_dict_ftunet_3 = model_load(checkpoint_ftunet_3['state_dict'], model_ftunet_3.state_dict())
    model_ftunet_3.load_state_dict(model_dict_ftunet_3)
    # model_ftunet_3 = load_checkpoint('model_weights/flair2/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_3/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_3.ckpt', model_ftunet_3)
    model_ftunet_3.cuda(config.gpus[0])
    model_ftunet_3.eval()

    model_ftunet_4 = ft_unetformer(num_classes=13, decoder_channels=256)
    checkpoint_ftunet_4 = torch.load('model_weights/flair2/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_4-Fake10/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_4-Fake10.ckpt', map_location={'cuda:4':'cuda:7'})
    model_dict_ftunet_4 = model_load(checkpoint_ftunet_4['state_dict'], model_ftunet_4.state_dict())
    model_ftunet_4.load_state_dict(model_dict_ftunet_4)
    # model_ftunet_4 = load_checkpoint('model_weights/flair2/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_4/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_4.ckpt', model_ftunet_4)
    model_ftunet_4.cuda(config.gpus[0])
    model_ftunet_4.eval()

    model_ftunet_5 = ft_unetformer(num_classes=13, decoder_channels=256)
    checkpoint_ftunet_5 = torch.load('model_weights/flair2/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_5-Fake10/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_5-Fake10.ckpt', map_location={'cuda:5':'cuda:7'})
    model_dict_ftunet_5 = model_load(checkpoint_ftunet_5['state_dict'], model_ftunet_5.state_dict())
    model_ftunet_5.load_state_dict(model_dict_ftunet_5)
    # model_ftunet_5 = load_checkpoint('model_weights/flair2/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_5/FT-JLoss-H_V_RR_O-Epoch40-RGB-IMS-K5_5.ckpt', model_ftunet_5)
    model_ftunet_5.cuda(config.gpus[0])
    model_ftunet_5.eval()




    # model_ftunet_2 = ft_unetformer(num_classes=13, decoder_channels=256)
    # model_ftunet_2 = load_checkpoint('model_weights/flair2/FT-Epoch40-FakeLabel-Epoch15-NRG-IMS/FT-Epoch40-FakeLabel-Epoch15-NRG-IMS.ckpt', model_ftunet_2)
    # model_ftunet_2.cuda(config.gpus[0])
    # model_ftunet_2.eval()

    # model_dc_1 = dcswin_small(num_classes=13)
    # model_dc_1 = load_checkpoint('model_weights/flair2/DC-Epoch30-FakeLabel-Epoch15-NRG-IMS/DC-Epoch30-FakeLabel-Epoch15-NRG-IMS.ckpt', model_dc_1)
    # model_dc_1.cuda(config.gpus[0])
    # model_dc_1.eval()

    # model_dl_1 = DeepLabv3_plus(nInputChannels=3, n_classes=13, os=16, pretrained=True, _print=True)
    # model_dl_1 = load_checkpoint('model_weights/flair2/DL-Epoch30-FakeLabel-Epoch15-NRG-IMS/DL-Epoch30-FakeLabel-Epoch15-NRG-IMS.ckpt', model_dl_1)
    # model_dl_1.cuda(config.gpus[0])
    # model_dl_1.eval()

    # model_un_1 = UNetFormer(num_classes=13)
    # model_un_1 = load_checkpoint('model_weights/flair2/UN-Epoch30-FakeLabel-Epoch15-NRG-IMS/UN-Epoch30-FakeLabel-Epoch15-NRG-IMS.ckpt', model_un_1)
    # model_un_1.cuda(config.gpus[0])
    # model_un_1.eval()

    # model_unet = UNetFormer(num_classes=13)
    # checkpoint_unet = torch.load('model_weights/flair2/UN-Epoch30-FakeLabel/UN-Epoch30-FakeLabel.ckpt', map_location={'cuda:3':'cuda:3'})
    # model_dict_unet = model_load(checkpoint_unet['state_dict'], model_unet.state_dict())
    # model_unet.load_state_dict(model_dict_unet)
    # model_unet.cuda(config.gpus[0])
    # model_unet.eval()

    # model_ftunet = ft_unetformer(num_classes=13, decoder_channels=256)
    # checkpoint_ftunet = torch.load('model_weights/flair2/FT-Epoch30-FakeLabel/FT-Epoch30-FakeLabel.ckpt', map_location={'cuda:3':'cuda:3'})
    # model_dict_ftunet = model_load(checkpoint_ftunet['state_dict'], model_ftunet.state_dict())
    # model_ftunet.load_state_dict(model_dict_ftunet)
    # model_ftunet.cuda(config.gpus[0])
    # model_ftunet.eval()

    # model_dl = DeepLabv3_plus(nInputChannels=3, n_classes=13, os=16, pretrained=True, _print=True)
    # checkpoint_dl = torch.load('model_weights/flair2/DL-Epoch30-FakeLabel/DL-Epoch30-FakeLabel.ckpt', map_location={'cuda:3':'cuda:3'})
    # model_dict_dl = model_load(checkpoint_dl['state_dict'], model_dl.state_dict())
    # model_dl.load_state_dict(model_dict_dl)
    # model_dl.cuda(config.gpus[0])
    # model_dl.eval()

    # model_dc = dcswin_small(num_classes=13)
    # checkpoint_dc = torch.load('model_weights/flair2/DC-Epoch30-FakeLabel/DC-Epoch30-FakeLabel.ckpt', map_location={'cuda:3':'cuda:3'})
    # model_dict_dc = model_load(checkpoint_dc['state_dict'], model_dc.state_dict())
    # model_dc.load_state_dict(model_dict_dc)
    # model_dc.cuda(config.gpus[0])
    # model_dc.eval()

    # model_up = UnetPlusPlus(num_classes=13)
    # checkpoint_up = torch.load('model_weights/flair/UNet-V1-FakeLabel/UNet-V1-FakeLabel.ckpt', map_location={'cuda:2':'cuda:3'})
    # model_dict_up = model_load(checkpoint_up['state_dict'], model_up.state_dict())
    # model_up.load_state_dict(model_dict_up)
    # model_up.cuda(config.gpus[0])
    # model_up.eval()
    
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

        # model_un_1 = tta.SegmentationTTAWrapper(model_un_1, transforms)
        # model_ftunet_1 = tta.SegmentationTTAWrapper(model_ftunet_1, transforms)
        # model_ftunet_2 = tta.SegmentationTTAWrapper(model_ftunet_2, transforms)
        # model_dl_1 = tta.SegmentationTTAWrapper(model_dl_1, transforms)
        # model_dc_1 = tta.SegmentationTTAWrapper(model_dc_1, transforms)
        # model_up = tta.SegmentationTTAWrapper(model_up, transforms)

        model_ftunet_1 = tta.SegmentationTTAWrapper(model_ftunet_1, transforms)
        model_ftunet_2 = tta.SegmentationTTAWrapper(model_ftunet_2, transforms)
        model_ftunet_3 = tta.SegmentationTTAWrapper(model_ftunet_3, transforms)
        model_ftunet_4 = tta.SegmentationTTAWrapper(model_ftunet_4, transforms)
        model_ftunet_5 = tta.SegmentationTTAWrapper(model_ftunet_5, transforms)

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

            # raw_predictions_unet = model_un_1(input['img'].cuda(config.gpus[0]))
            # raw_predictions_ftunet_1 = model_ftunet_1(input['img'].cuda(config.gpus[0]))
            # raw_predictions_ftunet_2 = model_ftunet_2(input['img'].cuda(config.gpus[0]))
            # raw_predictions_dl = model_dl_1(input['img'].cuda(config.gpus[0]))
            # raw_predictions_dc = model_dc_1(input['img'].cuda(config.gpus[0]))
            # raw_predictions_up = model_up(input['img'].cuda(config.gpus[0]))

            raw_predictions_ftunet_1 = model_ftunet_1(input['img'].cuda(config.gpus[0]))
            raw_predictions_ftunet_2 = model_ftunet_2(input['img'].cuda(config.gpus[0]))
            raw_predictions_ftunet_3 = model_ftunet_3(input['img'].cuda(config.gpus[0]))
            raw_predictions_ftunet_4 = model_ftunet_4(input['img'].cuda(config.gpus[0]))
            raw_predictions_ftunet_5 = model_ftunet_5(input['img'].cuda(config.gpus[0]))
            raw_predictions = raw_predictions_ftunet_1+raw_predictions_ftunet_2+raw_predictions_ftunet_3+raw_predictions_ftunet_4+raw_predictions_ftunet_5

            image_ids = input["name_id"]

            # raw_predictions = (raw_predictions_dl + raw_predictions_dc+ raw_predictions_unet
            #                     + raw_predictions_ftunet_2)

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(predictions.shape[0]):
                # mask_name = image_ids[i]
                # output_mask = np.zeros(shape=(512, 512, 3), dtype=np.uint8)

                # mask = predictions[i].cpu().numpy()
                # easy_mask = mask[np.newaxis, :, :]
                # output_mask[np.all(easy_mask == 0, axis=0)] = [0,0,0]
                # output_mask[np.all(easy_mask == 1, axis=0)] = [1,1,1]
                # output_mask[np.all(easy_mask == 2, axis=0)] = [2,2,2]
                # output_mask[np.all(easy_mask == 3, axis=0)] = [3,3,3]
                # output_mask[np.all(easy_mask == 4, axis=0)] = [4,4,4]
                # output_mask[np.all(easy_mask == 5, axis=0)] = [5,5,5]
                # output_mask[np.all(easy_mask == 6, axis=0)] = [6,6,6]
                # output_mask[np.all(easy_mask == 7, axis=0)] = [7,7,7]
                # output_mask[np.all(easy_mask == 8, axis=0)] = [8,8,8]
                # output_mask[np.all(easy_mask == 9, axis=0)] = [9,9,9]
                # output_mask[np.all(easy_mask == 10, axis=0)] = [10,10,10]
                # output_mask[np.all(easy_mask == 11, axis=0)] = [11,11,11]
                # output_mask[np.all(easy_mask == 12, axis=0)] = [12,12,12]

                # predictions_unet_hard = predictions_hard[i].cpu().numpy()
                
                # hard_mask = predictions_unet_hard[np.newaxis, :, :]
                # output_mask[np.all(hard_mask == 0, axis=0)] = [9,9,9]
                # output_mask[np.all(hard_mask == 1, axis=0)] = [10,10,10]
                # output_mask[np.all(hard_mask == 2, axis=0)] = [11,11,11]

                # output_mask = output_mask[:,:,0]
                # results.append((output_mask, str(args.output_path / mask_name)))

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