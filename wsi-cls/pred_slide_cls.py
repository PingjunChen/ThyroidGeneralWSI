# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
import time
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from wsinet import WsiNet

from patch_utils import split_regions, gen_slide_feas


def load_path_model(args):
    # load patch model
    patch_model_path = os.path.join(args.data_dir, "Models/PatchModels", args.model_type,
                                    args.model_session, args.patch_model_name)
    patch_model = torch.load(args.patch_model_path)
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


def test_cls(net, dataloader):
    start_timer = time.time()
    total_pred, total_gt = [], []
    for ind, (batch_feas, gt_classes, true_num, bboxes) in enumerate(dataloader):
        # print("Pred {:03d}/{:03d}".format(ind+1, len(dataloader)))
        im_data = Variable(batch_feas.cuda())
        cls_probs, assignments = net(im_data, None, true_num=true_num)
        _, cls_labels = torch.topk(cls_probs.cpu(), 1, dim=1)
        cls_labels = cls_labels.numpy()[:, 0]
        total_gt.extend(gt_classes.tolist())
        total_pred.extend(cls_labels.tolist())

    con_mat = confusion_matrix(total_gt, total_pred)
    cur_eval_acc = np.trace(con_mat) * 1.0 / np.sum(con_mat)

    total_time = time.time()-start_timer
    print("Testing Acc: {:.3f}".format(cur_eval_acc))
    print("Confusion matrix:")
    print(con_mat)


def set_args():
    parser = argparse.ArgumentParser(description = 'Thyroid WSI diagnois')
    parser.add_argument("--batch_size",      type=int,   default=24,      help="batch size")
    parser.add_argument('--device_id',       type=str,   default="4",     help='which device')
    parser.add_argument('--test_num',        type=int,   default=128,     help='which device')

    # model setting
    parser.add_argument("--class_num",       type=int,   default=3)
    parser.add_argument("--input_fea_num",   type=int,   default=4096)
    parser.add_argument("--data_dir",        type=str,   default="../data")
    parser.add_argument('--model_type',      type=str,   default="vgg16bn")
    parser.add_argument("--mode",            type=str,   default="pooling")
    parser.add_argument('--wsi_cls_name',    type=str,   default="model-816.pth")

    parser.add_argument('--model_dir',       type=str, default="../data/Models/PatchModels")
    parser.add_argument('--model_session',   type=str, default="01")
    parser.add_argument('--patch_model_name',type=str, default="thyroid02-0.7751.pth")

    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--img_level',       type=int,   default=2)
    parser.add_argument('--cnt_level',       type=int,   default=3)

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
    print(">> START overlaying...")
    for ind, cur_slide in enumerate(slide_list):
        cur_slide_path = os.path.join(test_slide_dir, cur_slide)
        split_arr, patch_list, wsi_dim, s_img, mask = split_regions(cur_slide_path, args.img_level, args.cnt_level)
        if len(split_arr) == 0:
            continue
        probs, logits, feas, bboxes = gen_slide_feas(cls_model, split_arr, np.asarray(patch_list), wsi_dim, args)


    # test_cls(wsi_net, test_dataloader)
