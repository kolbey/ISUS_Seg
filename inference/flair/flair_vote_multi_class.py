'''
对多个模型的预测保存的结果进行投票融合，
具体方案是相同位置的类别值采用少数服从多数的形式确定！！
'''
import numpy as np
import os
from PIL import Image
import cv2
from tqdm import tqdm

import multiprocessing.pool as mpp
import multiprocessing as mp
import time

def check_dir(target_dir):
    if not os.path.isdir(target_dir):
        print('Create file directory => %s' % target_dir)
        os.makedirs(target_dir)

def load_mask_name(path):
    mask_list = sorted(os.listdir(path))
    return mask_list

# def vote_multi_class(MASK_PATH, mask_name, output_path, height, width):
def vote_multi_class(mask_name):

    MASK_PATH = ['./fig_results/flair2/ft-epoch40-fakelabel-epoch15-nrg-ims/',
                './fig_results/flair2/dc-epoch30-fakelabel-epoch15-nrg-ims/',
                './fig_results/flair2/ft-epoch40-fakelabel-epoch15-nrg/',
                './fig_results/flair2/ft-epoch40-fakelabel-epoch15-ndrg/',
                './fig_results/flair2/dc-epoch30-fakelabel/',
                './fig_results/flair2/un-ft-dl-dc-epoch30/',
                './fig_results/flair2/dl-epoch30-fakelabel-epoch15-nrg-ims/']
    height = 512
    width = 512
    output_path = './fig_results/flair2/hard_vote_model7'

    mask_list = []
    for j in range(len(MASK_PATH)):
        mask = np.array(Image.open(MASK_PATH[j]+str(mask_name)))
        mask_list.append(mask)

    vote_mask = np.zeros((height, width)).astype(np.uint8)
    for h in range(height):
        for w in range(width):
            scores = np.zeros((1,13)).astype(np.uint8)
            for i in range(len(mask_list)):
                mask = mask_list[i]
                pixel = mask[h,w]
                scores[0, pixel] += 1

            label_id = scores.argmax()
            vote_mask[h,w] = label_id

    output_path = output_path + '/' + str(mask_name)
    cv2.imwrite(output_path, vote_mask)


if __name__ == '__main__':

    input_path = './fig_results/flair2/ft-epoch40-fakelabel-epoch15-nrg-ims/'
    output_path = './fig_results/flair2/hard_vote_model7'

    check_dir(output_path)

    results_list = load_mask_name(input_path)

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(vote_multi_class, results_list)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))

