# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from skimage import io
import pydaily

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from wsinet import WsiNet
from patch_utils import split_regions, gen_slide_feas
from patch_utils import overlayWSI


def load_path_model(args):
    # load patch model
    patch_model_path = os.path.join(args.data_dir, "Models/PatchModels", args.model_type,
                                    args.model_session, args.patch_model_name)
    patch_model = torch.load(patch_model_path)
    patch_model.cuda()
    patch_model.eval()

    return patch_model


def load_wsinet(args):
    wsinet = WsiNet(class_num=args.class_num, in_channels=args.input_fea_num, mode=args.mode)
    weightspath = os.path.join(args.data_dir, "Models/SlideModels/BestModels", args.model_type,
                               args.mode, args.wsi_cls_name)
    wsi_weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    wsinet.load_state_dict(wsi_weights_dict)
    wsinet.cuda()
    wsinet.eval()

    return wsinet


def set_args():
    parser = argparse.ArgumentParser(description = 'Thyroid WSI diagnois')
    parser.add_argument('--device_id',       type=str,   default="5")
    parser.add_argument('--test_num',        type=int,   default=128)

    # model setting
    parser.add_argument("--class_num",       type=int,   default=3)
    parser.add_argument("--input_fea_num",   type=int,   default=4096)
    parser.add_argument("--data_dir",        type=str,   default="../data")
    parser.add_argument('--model_type',      type=str,   default="vgg16bn")
    parser.add_argument("--mode",            type=str,   default="selfatt")
    parser.add_argument('--wsi_cls_name',    type=str,   default="model-838.pth")

    parser.add_argument('--model_dir',       type=str,   default="../data/Models/PatchModels")
    parser.add_argument('--model_session',   type=str,   default="01")
    parser.add_argument('--patch_model_name',type=str,   default="thyroid02-0.7751.pth")

    parser.add_argument('--patch_size',      type=int,   default=224)
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--img_level',       type=int,   default=2)
    parser.add_argument('--cnt_level',       type=int,   default=3)

    parser.add_argument('--save_overlay',    action='store_true', default=True)
    parser.add_argument('--verbose',         action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    # model
    patch_model = load_path_model(args)
    wsi_model = load_wsinet(args)
    # dataset
    test_slide_dir = os.path.join(args.data_dir, "Slides/test/3Malignant")
    slide_list = [ele for ele in os.listdir(test_slide_dir) if "tiff" in ele]
    print(">> START diagnosis...")
    if args.save_overlay == True:
        overlay_save_dir = os.path.join(os.path.dirname(test_slide_dir), "overlay"+str(args.test_num))
        pydaily.filesystem.overwrite_dir(overlay_save_dir)

    correct_num = 0
    since = time.time()
    for ind, cur_slide in enumerate(slide_list):
        cur_slide_path = os.path.join(test_slide_dir, cur_slide)
        split_arr, patch_list, wsi_dim, s_img, mask = split_regions(cur_slide_path, args.img_level, args.cnt_level)
        if len(split_arr) == 0:
            continue
        probs, logits, feas, bboxes = gen_slide_feas(patch_model, split_arr, np.asarray(patch_list), wsi_dim, args)
        feas_tensor = torch.from_numpy(feas).unsqueeze(0)
        feas_data = Variable(feas_tensor.cuda())
        cls_probs, assignments = wsi_model(feas_data)
        _, cls_labels = torch.topk(cls_probs.cpu(), 1, dim=1)
        cur_label = cls_labels.numpy()[0, 0]
        print("{} prediction is : {}".format(cur_slide, cur_label))

        if args.save_overlay == True and cur_label == 2:
            weights = assignments[0].data.cpu().tolist()
            overlay_wsi = overlayWSI(cur_slide_path, bboxes, weights, args.img_level)
            overlay_wsi_name = os.path.splitext(cur_slide)[0] + ".png"
            io.imsave(os.path.join(overlay_save_dir, overlay_wsi_name), overlay_wsi)

    time_elapsed = time.time() - since
    print("Testing takes {:.0f}m {:.2f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Testing accuracy is {}/{}".format(correct_num, len(slide_names)))
