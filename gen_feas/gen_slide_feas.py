# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import argparse, time
import torch
from pydaily import filesystem
from pyimg import combine
import openslide
import matplotlib.pyplot as plt
from skimage import io, transform
import deepdish as dd
import utils, patch_util


def predict_slide_fea(slide_path, cls_model, save_dir, args):
    file_fullname = os.path.basename(slide_path)
    file_name = os.path.splitext(file_fullname)[0]
    file_cat = os.path.basename(os.path.dirname(slide_path))
    fea_save_dir = os.path.join(save_dir, file_cat)
    if not os.path.exists(fea_save_dir):
        os.makedirs(fea_save_dir)
    fea_filepath = os.path.join(fea_save_dir, file_name + ".h5")

    # print("Step 1: Split slide to patches")
    split_arr, patch_list, wsi_dim, s_img, mask = utils.split_regions(
        slide_path, args.img_level, args.cnt_level)
    if len(split_arr) == 0:
        return None

    # # save mask overlay image to validate the accuracy of tissue localization
    # mask_overlay = combine.blend_images(s_img, combine.graymask2rgb(mask), alpha=0.64)
    # s_mask_overlay = transform.resize(mask_overlay, (int(mask.shape[0]*0.3), int(mask.shape[1]*0.3)))
    # io.imsave(os.path.join(save_dir, file_name + ".png"), s_mask_overlay)

    # print("Step 2: Generate features")
    fea_dict = patch_util.gen_slide_feas(cls_model, split_arr, np.asarray(patch_list), wsi_dim, args)
    # save features
    dd.io.save(fea_filepath, fea_dict)


def predit_all_feas(model, args):
    slide_path = os.path.join(args.slide_dir, args.dset)
    slide_list = filesystem.find_ext_files(slide_path, "tiff")
    print("There are {} slides in totoal.".format(len(slide_list)))
    slide_list.sort()

    fea_dir = os.path.join(args.fea_dir, args.model_type, args.dset)

    print("Start processing...")
    print("="*80)
    slide_start = time.time()
    for ind, slide_path in enumerate(slide_list):
        slide_filename = os.path.splitext(os.path.basename(slide_path))[0]
        slide_head = openslide.OpenSlide(slide_path)
        print("Processing {}, width: {}, height: {}, {}/{}".format(
            slide_filename, slide_head.dimensions[0], slide_head.dimensions[1], ind+1, len(slide_list)))
        predict_slide_fea(slide_path, model, fea_dir, args)
    print("="*80)
    slide_elapsed = time.time() - slide_start
    print("Time cost: " + time.strftime("%H:%M:%S", time.gmtime(slide_elapsed)))
    print("Finish Prediction...")


def set_args():
    parser = argparse.ArgumentParser(description="Settings for thyroid slide patch feature generation")
    parser.add_argument('--device_id',            type=str, default="5",     help='which device')
    parser.add_argument('--slide_dir',            type=str, default="../data/CV01/Slides")
    parser.add_argument('--fea_dir',              type=str, default="../data/CV01/Feas")
    parser.add_argument('--dset',                 type=str, default="test")
    # patch model setting
    parser.add_argument('--model_dir',            type=str, default="../data/CV01/Models/PatchModels")
    parser.add_argument('--model_type',           type=str, default="resnet18")
    parser.add_argument('--model_name',           type=str, default="thyroid00-0.7829.pth")
    parser.add_argument('--patch_size',           type=int, default=224)
    parser.add_argument('--batch_size',           type=int, default=64)
    parser.add_argument('--img_level',            type=int, default=2)
    parser.add_argument('--cnt_level',            type=int, default=3)
    parser.add_argument('--verbose',              action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # load patch model
    args.model_path = os.path.join(args.model_dir, args.model_type, args.model_name)
    if not os.path.exists(args.model_path):
        raise AssertionError("Model path does not exist")
    ft_model = torch.load(args.model_path)
    ft_model.cuda()
    ft_model.eval()

    # predict all patches
    print("Prediction model is: {}".format(args.model_name))
    predit_all_feas(model=ft_model, args=args)
