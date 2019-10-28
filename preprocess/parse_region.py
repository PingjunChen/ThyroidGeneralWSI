# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import openslide
import deepdish as dd
from pydaily import format


def save_wsi_annotation(slide_path, json_path, h5_path, slide_level):
    slide_head = openslide.OpenSlide(slide_path)
    region_dict = {}
    annotation_dict = format.json_to_dict(json_path)
    regions = annotation_dict['regions']
    for region_id in regions.keys():
        region_name = 'r' + str(region_id)
        cur_region = {}

        if regions[region_id]['desp'] == "Benign":
            cur_region['desp'] = "1Benign"
        elif regions[region_id]['desp'] == "Uncertain":
            cur_region['desp'] = "2Uncertain"
        elif regions[region_id]['desp'] == "Malignant":
            cur_region['desp'] = "3Malignant"
        else:
            print("Unknow description: {}".format(regions[region_id]['desp']))
            continue

        cur_cnts = regions[region_id]['cnts']
        num_points = len(cur_cnts["h"])
        points_coors = np.zeros((2, num_points), dtype=np.int32)
        for ind in range(num_points):
            points_coors[0, ind] = int(round(cur_cnts['h'][ind] / np.power(2, slide_level)))
            points_coors[1, ind] = int(round(cur_cnts['w'][ind] / np.power(2, slide_level)))
        cur_region['cnts'] = points_coors
        start_h, start_w = np.min(points_coors[0, :]), np.min(points_coors[1, :])
        region_h = np.max(points_coors[0, :]) - start_h + 1
        region_w = np.max(points_coors[1, :]) - start_w + 1
        region_img = slide_head.read_region(location=(start_w, start_h), level=slide_level,
                                            size=(region_w, region_h))
        region_img = np.asarray(region_img)[:, :, :3]
        cur_region['img'] = region_img
        region_dict[region_name] = cur_region
    dd.io.save(h5_path, region_dict)


def parse_all_annotations(data_dir, cur_set, cur_cat, slide_level):
    slides_dir = os.path.join(data_dir, "Slides", cur_set, cur_cat)
    annotations_dir = os.path.join(data_dir, "Annotations", cur_set, cur_cat)
    regionH5_save_dir = os.path.join(data_dir, "Regions", cur_set, cur_cat)
    if not os.path.exists(regionH5_save_dir):
        os.makedirs(regionH5_save_dir)

    slides_list = [ele for ele in os.listdir(slides_dir) if "tiff" in ele]
    for ele in slides_list:
        slide_name = os.path.splitext(ele)[0]
        slide_path = os.path.join(slides_dir, ele)
        json_path = os.path.join(annotations_dir, slide_name+".json")
        h5_path = os.path.join(regionH5_save_dir, slide_name+".h5")

        if os.path.exists(h5_path):
            continue
        save_wsi_annotation(slide_path, json_path, h5_path, slide_level=slide_level)



if __name__ == '__main__':
    np.random.seed(3333)

    data_dir = "../data/CV04"
    categories = ["1Benign", "2Uncertain", "3Malignant"]
    which_set = ["train", "val"]

    for cur_set in which_set:
        print("Current set is: {}".format(cur_set))
        for cur_cat in categories:
            print("Current category is: {}".format(cur_cat))
            parse_all_annotations(data_dir, cur_set, cur_cat, slide_level=2)
