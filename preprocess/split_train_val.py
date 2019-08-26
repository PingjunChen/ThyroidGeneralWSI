# -*- coding: utf-8 -*-

import os, sys
from random import shuffle
import shutil

def split_valset(patch_dir, ratio):
    train_dir = os.path.join(patch_dir, "Train")
    cats_list = os.listdir(train_dir)
    for cur_cat in cats_list:
        cur_train_dir = os.path.join(train_dir, cur_cat)
        file_list = [ele for ele in os.listdir(cur_train_dir) if "png" in ele]
        shuffle(file_list)
        val_num = int(len(file_list) * ratio)
        for ind in range(val_num):
            shutil.move(os.path.join(cur_train_dir, file_list[ind]),
                        os.path.join(patch_dir, "Val", cur_cat))

if __name__ == "__main__":
    patch_dir = "../data/Patches"
    split_valset(patch_dir, 0.1)
