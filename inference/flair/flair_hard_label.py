'''
个人认为二分类模型性能更好，
故用二分类预测结果对多分类的预测进行矫正,
实际情况不理想！
'''
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
from PIL import Image
import os
import os.path as osp
import time

def vote_hard_label(inp):
    (output_path, result_fake, result_hard) = inp
    fake_list = sorted(os.listdir(result_fake))
    hard_list = sorted(os.listdir(result_hard))
    assert len(fake_list) == len(hard_list)

    for i in range(len(fake_list)):
        fake_label_name = fake_list[i]
        hard_label_name = hard_list[i]
        assert str(fake_label_name) == str(hard_label_name)

        fake_label_path = osp.join(result_fake, fake_label_name)
        hard_label_path = osp.join(result_hard, hard_label_name)

        fake_label = np.array(Image.open(fake_label_path))
        hard_label = np.array(Image.open(hard_label_path))

        height, width = fake_label.shape

        for h in range(height):
            for w in range(width):
                if hard_label[h][w] == 0:
                    fake_label[h][w] = 4

        output_label = fake_label.astype(np.uint8)
        mask_name_png = output_path + fake_label_name
        cv2.imwrite(mask_name_png, output_label)


result_fake = './fig_results/flair/mm-un-ft-dl-dc-fake-v2-up0/'
result_hard = './fig_results/flair/un-v1-hardlabel-4/'
output_path = './fig_results/flair/mm-un-ft-dl-dc-fake-v2-up0-4/'

results = []
results.append((output_path, result_fake, result_hard))
t0 = time.time()
mpp.Pool(processes=mp.cpu_count()).map(vote_hard_label, results)
t1 = time.time()
img_write_time = t1 - t0
print('images writing spends: {} s'.format(img_write_time))