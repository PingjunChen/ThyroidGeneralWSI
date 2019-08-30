# -*- coding: utf-8 -*-

import os, sys
import deepdish as dd
import argparse
from pydaily import filesystem


def set_args():
    parser = argparse.ArgumentParser(description="check fea generation")
    parser.add_argument('--fea_dir',              type=str, default="../data/Feas/resnet50")
    parser.add_argument('--dset',                 type=str, default="test")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_args()
    feas_dir = os.path.join(args.fea_dir, args.dset)
    h5_list = filesystem.find_ext_files(feas_dir, ".h5")
    for ele in h5_list:
        cur_fea_dict = dd.io.load(ele)
        num_patch = len(cur_fea_dict['prob'])
        if num_patch <= 128:
            print("Regions in {} is {}".format(ele, num_patch))
