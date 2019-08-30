# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import openslide
import cv2
from skimage import io
from pydaily import filesystem
from pydaily import format
from pyimg import combine



def annotate_images(data_dir, cur_set, cur_cat, slide_level):
    slides_dir = os.path.join(data_dir, "Slides", cur_set, cur_cat)
    slide_list = [ele for ele in os.listdir(slides_dir) if "tiff" in ele]

    annotation_dir = os.path.join(data_dir, "Annotations", cur_set, cur_cat)
    l2_annotate_dir = os.path.join(data_dir, "L4AnnotatedImgs", cur_set, cur_cat)
    filesystem.overwrite_dir(l2_annotate_dir)

    for cur_slide in slide_list:
        slide_name = os.path.splitext(cur_slide)[0]
        slide_head = openslide.OpenSlide(os.path.join(slides_dir, cur_slide))
        slide_img = slide_head.read_region(location=(0, 0), level=slide_level,
                                           size=slide_head.level_dimensions[slide_level])
        slide_img = np.asarray(slide_img)[:, :, :3]

        annotate_json_path =  os.path.join(annotation_dir, slide_name+".json")
        annotation_dict = format.json_to_dict(annotate_json_path)
        region_dict = annotation_dict['regions']
        for key in region_dict.keys():
            cur_label = region_dict[key]['desp']
            draw_rgb = None
            if cur_label == "Benign":
                draw_rgb = (0, 0, 255)
            elif cur_label == "Uncertain":
                draw_rgb = (0, 255, 0)
            elif cur_label == "Malignant":
                draw_rgb = (255, 0, 0)
            else:
                print("Unknow description: {}".format(cur_label))
                continue

            cur_cnts = region_dict[key]['cnts']
            num_points = len(cur_cnts["h"])
            points_coors = np.zeros((2, num_points), dtype=np.int32)
            for ind in range(num_points):
                points_coors[0, ind] = int(round(cur_cnts['h'][ind] / np.power(2, slide_level)))
                points_coors[1, ind] = int(round(cur_cnts['w'][ind] / np.power(2, slide_level)))
            slide_img = combine.overlay_contour(slide_img, points_coors, draw_rgb, cnt_width=5)
            tl_pos = (int(np.mean(points_coors[0])), int(np.mean(points_coors[1])))
            cv2.putText(slide_img, cur_label, tl_pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (148,24,32), 3, cv2.LINE_AA)
        annotate_slide_path = os.path.join(l2_annotate_dir, slide_name+".png")
        io.imsave(annotate_slide_path, slide_img)



if __name__ == "__main__":
    np.random.seed(3333)

    slide_level=4
    data_dir = "../data/TrainVal"
    categories = ["1Benign", "2Uncertain", "3Malignant"]
    which_set = ["val", "train"]

    for cur_set in which_set:
        print("Current set is: {}".format(cur_set))
        for cur_cat in categories:
            print("Current category is: {}".format(cur_cat))
            annotate_images(data_dir, cur_set, cur_cat, slide_level=slide_level)
