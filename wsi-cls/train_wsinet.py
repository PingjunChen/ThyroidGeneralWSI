# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pydaily import filesystem

from thyroid_dataset import ThyroidDataSet
from wsinet  import WsiNet
from train_eng import train_cls


def set_args():
    parser = argparse.ArgumentParser(description = 'WSI diagnois')
    parser.add_argument("--batch_size",      type=int,   default=24,      help="batch size")
    parser.add_argument("--lr",              type=float, default=1.0e-2,  help="learning rate (default: 0.01)")
    parser.add_argument("--momentum",        type=float, default=0.9,     help="SGD momentum (default: 0.5)")
    parser.add_argument("--weight_decay",    type=float, default=5.0e-4,  help="weight decay for training")
    parser.add_argument("--maxepoch",        type=int,   default=100,     help="number of epochs to train")
    parser.add_argument("--decay_epoch",     type=int,   default=1,       help="lr start to decay linearly from decay_epoch")
    parser.add_argument("--save_freq",       type=int,   default=1,       help="how frequent to save the model")

    # model setting
    parser.add_argument('--device_id',       type=str,   default="7",     help='which device')
    parser.add_argument("--data_dir",        type=str,   default="../data/CV02")
    parser.add_argument('--model_type',      type=str,   default="resnet18")
    parser.add_argument("--input_fea_num",   type=int,   default=512)
    parser.add_argument("--mode",            type=str,   default="selfatt")
    parser.add_argument("--class_num",       type=int,   default=3)

    parser.add_argument("--pre_load",        action='store_true', default=True)
    parser.add_argument('--verbose',         action='store_true')
    args = parser.parse_args()
    return args


if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # Model preparetion
    net = WsiNet(class_num=args.class_num, in_channels=args.input_fea_num, mode=args.mode)
    net.cuda()

    # Dataset preparetion
    train_data_root = os.path.join(args.data_dir, "Feas", args.model_type, "train")
    val_data_root = os.path.join(args.data_dir, "Feas", args.model_type, "val")
    # create dataset
    train_dataset = ThyroidDataSet(train_data_root, testing=False, pre_load=args.pre_load)
    val_dataset = ThyroidDataSet(val_data_root, testing=True, testing_num=128, pre_load=args.pre_load)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, num_workers=4, pin_memory=True)

    print(">> START training")
    model_root = os.path.join(args.data_dir, "Models", "SlideModels", args.model_type, args.mode)
    filesystem.overwrite_dir(model_root)
    train_cls(train_dataloader, val_dataloader, model_root, net, args)
