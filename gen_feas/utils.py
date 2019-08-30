# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import openslide
from skimage import color, filters
from skimage import img_as_ubyte
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from shapely.geometry import Polygon
import cv2
import math


# Locate tissue regions from kfb low level image
def find_tissue_cnt(slide_path, level=3, thresh_val=0.82, min_size=2.0e5):
    slide_head = openslide.OpenSlide(slide_path)
    wsi_img = slide_head.read_region((0, 0), level, slide_head.level_dimensions[level])
    wsi_img = np.array(wsi_img)[:,:,:3]
    # Gray
    gray = color.rgb2gray(wsi_img)
    # Smooth
    smooth = filters.gaussian(gray, sigma=9)
    # Threshold
    binary = smooth < thresh_val
    # Fill holes
    fill = binary_fill_holes(binary)
    # Remove outliers
    mask = remove_small_objects(fill, min_size=min_size, connectivity=8)
    # Find contours
    _, cnts, _ = cv2.findContours(img_as_ubyte(mask),
                                  mode=cv2.RETR_EXTERNAL,
                                  method=cv2.CHAIN_APPROX_NONE)
    return wsi_img, mask, cnts


def split_regions(slide_path, img_level=2, cnt_level=3):
    s_img, mask, cnts = find_tissue_cnt(slide_path, cnt_level)
    img_cnt_ratio = 2**(cnt_level-img_level)
    wsi_dim = [ele*img_cnt_ratio for ele in s_img.shape[:2]]
    slide_head = openslide.OpenSlide(slide_path)
    wsi_img = slide_head.read_region((0, 0), img_level, wsi_dim)
    wsi_img = np.array(wsi_img)[:,:,:3]

    RAW_SIZE = 256
    SIZE1, SIZE2, SIZE4 = int(RAW_SIZE/4), int(RAW_SIZE/2), RAW_SIZE
    split_arr, patch_list = [], []
    for c_ind in range(len(cnts)):
        cur_cnt = cnts[c_ind] * img_cnt_ratio
        cur_cnt = np.squeeze(cur_cnt)
        w_coors = [int(round(ele)) for ele in cur_cnt[:, 0].tolist()]
        h_coors = [int(round(ele)) for ele in cur_cnt[:, 1].tolist()]
        w_min, w_max = min(w_coors), max(w_coors)
        h_min, h_max = min(h_coors), max(h_coors)

        # Width range to crop
        start_w = (w_min - SIZE1) if (w_min - SIZE1) > 0 else 0
        num_w = int(math.ceil((w_max - start_w - SIZE2)/SIZE2))
        # Height range to crop
        start_h = (h_min - SIZE1) if (h_min - SIZE1) > 0 else 0
        num_h = int(math.ceil((h_max - start_h - SIZE2)/SIZE2))

        poly_cnt = Polygon(cur_cnt)
        if poly_cnt.is_valid == False:
            continue
        for iw in range(0, num_w):
            for ih in range(0, num_h):
                # determine current rectangular is inside the contour or not
                cur_coors = [(start_w+iw*SIZE2, start_h+ih*SIZE2), (start_w+iw*SIZE2+SIZE4, start_h+ih*SIZE2),
                             (start_w+iw*SIZE2+SIZE4, start_h+ih*SIZE2+SIZE4), (start_w+iw*SIZE2, start_h+ih*SIZE2+SIZE4)]
                if start_w+iw*SIZE2+SIZE4 >= wsi_img.shape[1] or start_h+ih*SIZE2+SIZE4 > wsi_img.shape[0]:
                    continue
                patch_cnt = Polygon(cur_coors)
                try:
                    inter_flag = poly_cnt.intersects(patch_cnt)
                    if inter_flag == False:
                        continue
                    else:
                        inter_cnt = poly_cnt.intersection(patch_cnt)
                        if inter_cnt.area > patch_cnt.area * 0.5:
                            split_arr.append((start_h+ih*SIZE2, start_w+iw*SIZE2))
                            split_patch = wsi_img[start_h+ih*SIZE2:start_h+ih*SIZE2+SIZE4, start_w+iw*SIZE2:start_w+iw*SIZE2+SIZE4, :]
                            patch_list.append(split_patch)
                except:
                    print("Error in Polygon relationship")
    return split_arr, patch_list, wsi_dim, s_img, mask
