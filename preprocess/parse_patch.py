# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import deepdish as dd
from skimage import io, color
import uuid

from pycontour import rela
from pyslide import contour


def extract_patches(data_dir, cur_set, cur_cat, patch_size):
    region_dir = os.path.join(data_dir, "Regions", cur_set, cur_cat)
    patch_dir = os.path.join(data_dir, "Patches", cur_set)
    h5_list = [ele for ele in os.listdir(region_dir) if "h5" in ele]
    for cur_h5 in h5_list:
        h5_path = os.path.join(region_dir, cur_h5)
        region_dict = dd.io.load(h5_path)
        for ir in region_dict.keys():
            cur_region = region_dict[ir]
            cur_img = cur_region['img']
            cur_cnt = cur_region['cnts']
            min_h_coor, min_w_coor = np.min(cur_cnt[0, :]), np.min(cur_cnt[1, :])
            for ind in range(cur_cnt.shape[1]):
                cur_cnt[0, ind] -= min_h_coor
                cur_cnt[1, ind] -= min_w_coor

            max_rand_h = np.max(cur_cnt[0, :]) - patch_size - 1
            max_rand_w = np.max(cur_cnt[1, :]) - patch_size - 1
            if max_rand_h <= 0 or max_rand_w <= 0:
                continue

            cur_height, cur_width = cur_img.shape[0], cur_img.shape[1]
            height_ratio = int(np.ceil(cur_height / patch_size))
            width_ratio = int(np.ceil(cur_width / patch_size))

            # Number of patches to crop
            random_num = int(height_ratio * width_ratio / 4)

            for ind in range(random_num):
                rand_h = np.random.randint(0, max_rand_h)
                rand_w = np.random.randint(0, max_rand_w)
                cur_patch_coors = np.array([[rand_h, rand_h, rand_h+patch_size, rand_h+patch_size],
                                   [rand_w, rand_w+patch_size, rand_w+patch_size, rand_w]])

                inter_poly = rela.construct_intersection_polygon(cur_patch_coors, cur_cnt)
                if inter_poly == None:
                    continue
                inside_ratio = inter_poly.area / (patch_size * patch_size)

                if inside_ratio >= 0.75:
                    crop_img = cur_img[rand_h:rand_h+patch_size, rand_w:rand_w+patch_size, :]
                    if cur_region['desp'] == '2Uncertain' or cur_region['desp'] == '3Malignant':
                        g_img = color.rgb2gray(crop_img)
                        bk_num = (g_img > 0.72).sum()
                        bk_ratio = bk_num * 1.0 / (patch_size * patch_size)
                        if bk_ratio > 0.64:
                            continue
                    cur_path_dir = os.path.join(patch_dir, cur_region['desp'])
                    if not os.path.exists(cur_path_dir):
                        os.makedirs(cur_path_dir)
                    cur_patch_path = os.path.join(cur_path_dir, str(uuid.uuid4())[:8] + '.png')
                    io.imsave(cur_patch_path, crop_img)



if __name__ == '__main__':
    np.random.seed(1238)
    data_dir = "../data/CV04"
    # categories = ["1Benign", "2Uncertain", "3Malignant"]
    categories = ["2Uncertain", "3Malignant"]
    which_set = ["train", "val"]

    for cur_set in which_set:
        print("Current set is: {}".format(cur_set))
        for cur_cat in categories:
            print("Current category is: {}".format(cur_cat))
            extract_patches(data_dir, cur_set, cur_cat, patch_size=256)
