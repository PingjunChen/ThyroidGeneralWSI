# -*- coding: utf-8 -*-

import os, sys
import shutil
import tissueloc as tl
from tissueloc.load_slide import load_slide_img, select_slide_level
import numpy as np
from skimage import io, color
import cv2



if __name__ == "__main__":
    slide_dir = "../data/TestSlides/Malignant"
    save_dir = "../data/TestSlides/MalignantTissue"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    slide_list = [ele for ele in os.listdir(slide_dir) if "tiff" in ele]
    for ind, ele in enumerate(slide_list):
        slide_path = os.path.join(slide_dir, ele)
        cnts, d_factor = tl.locate_tissue_cnts(slide_path, max_img_size=2048, smooth_sigma=13,
                                               thresh_val=0.88,min_tissue_size=10000)
        s_level, d_factor = select_slide_level(slide_path, max_size=2048)
        slide_img = load_slide_img(slide_path, s_level)
        slide_img = np.ascontiguousarray(slide_img, dtype=np.uint8)
        cv2.drawContours(slide_img, cnts, -1, (0, 255, 0), 9)
        io.imsave(os.path.join(save_dir, os.path.join(os.path.splitext(ele)[0]+'_cnt.png')), slide_img)
