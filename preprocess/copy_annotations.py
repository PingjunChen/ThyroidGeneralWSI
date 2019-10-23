# -*- coding: utf-8 -*-

import os, sys
import shutil

if __name__ == "__main__":
    slides_dir = "../data/CV03/Slides"
    src_anno_dir = "../data/TrainAll/Annotations"
    dst_anno_dir = "../data/CV03/Annotations"

    categories = ["1Benign", "2Uncertain", "3Malignant"]
    splits = ["train", "val"]
    combs = [[x,y] for x in splits for y in categories]
    for comb in combs:
        split, cat = comb
        sub_slide_dir = os.path.join(slides_dir, split, cat)
        filenames = [ele[:-5] for ele in os.listdir(sub_slide_dir) if "tiff" in ele]
        for cur_name in filenames:
            src_anno_file = os.path.join(src_anno_dir, cat, cur_name+".json")
            cur_dst_anno_dir = os.path.join(dst_anno_dir, split, cat)
            if not os.path.exists(cur_dst_anno_dir):
                os.makedirs(cur_dst_anno_dir)
            shutil.copy(src_anno_file, cur_dst_anno_dir)
