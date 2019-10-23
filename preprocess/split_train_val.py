# -*- coding: utf-8 -*-

import os, sys
from sklearn.model_selection import train_test_split
import shutil


if __name__ == "__main__":
    split_seed = 1237
    src_dir = "../data/TrainAll/Slides"
    dst_dir = "../data/CV03/Slides"
    categories = ["1Benign", "2Uncertain", "3Malignant"]
    for cur_cat in categories:
        cur_cat_dir = os.path.join(src_dir, cur_cat)
        cur_cat_files = [ele for ele in os.listdir(cur_cat_dir) if "tiff" in ele]
        x_train, x_test = train_test_split(cur_cat_files, test_size=0.2, random_state=split_seed)
        data_dict = {"train": x_train, "val": x_test}
        for key, dlist in data_dict.items():
            for filename in dlist:
                src_file_path = os.path.join(cur_cat_dir, filename)
                dst_file_dir = os.path.join(dst_dir, key, cur_cat)
                if not os.path.exists(dst_file_dir):
                    os.makedirs(dst_file_dir)
                shutil.copy(src_file_path, dst_file_dir)
