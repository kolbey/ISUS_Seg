'''
将多通道的图像转变为 RGB 三通道的图像！
'''
import numpy as np
import os
from PIL import Image
import cv2
from tqdm import tqdm

import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import rasterio
from pathlib import Path

def check_dir(target_dir):
    if not os.path.isdir(target_dir):
        print('Create file directory => %s' % target_dir)
        os.makedirs(target_dir)

def load_img_name(path):
    def get_data_paths (path, filter):
                for path in Path(path).rglob(filter):
                    yield path.resolve().as_posix()

    domains = [Path(path, domain) for domain in os.listdir(path)]
    img_list = {'IMG':[]}
    for domain in domains:
         img_list['IMG'] += sorted(list(get_data_paths(domain, 'IMG*.tif')), 
                                   key=lambda x: int(x.split('_')[-1][:-4]))
         img_list['IMG'] = sorted(img_list['IMG'])
    return img_list['IMG']

def read_img(raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array


def visual_rgb(img_path):
    OUTPUT_PATH = './fig_results/crfs_test/flair_visual_rgb'
    img = read_img(img_path)
    img = img[:3, ...].transpose(1,2,0)

    img_name = os.path.basename(img_path)

    output_path = OUTPUT_PATH + '/' + img_name

    cv2.imwrite(output_path, img)

if __name__ == '__main__':
     input_path = '../flair/test'
     output_path = './fig_results/crfs_test/flair_visual_rgb'
     img_list = load_img_name(input_path)

     check_dir(output_path)

     t0 = time.time()
     mpp.Pool(processes=mp.cpu_count()).map(visual_rgb, img_list)
     t1 = time.time()
     img_write_time = t1 - t0
     print('images writing spends: {} s'.format(img_write_time))
