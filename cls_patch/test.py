# -*- coding: utf-8 -*-

import os, sys
import argparse
import torch
import torch.backends.cudnn as cudnn

from train_eng import validate_model
from data_ld import val_loader

def set_args():
    parser = argparse.ArgumentParser(description='Thyroid Classification')
    parser.add_argument('--seed',            type=int,   default=1234)
    parser.add_argument('--model_dir',       type=str,   default="../data/Models/PatchModels")
    parser.add_argument('--data_name',       type=str,   default="thyroid")
    parser.add_argument('--model_name',      type=str,   default="InceptionV3")
    parser.add_argument('--model_path',      type=str,   default="thyroid11-0.9731.pth")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args = set_args()
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    model_full_path = os.path.join(args.model_dir, args.model_name, "MixBest", args.model_path)
    ft_model = torch.load(model_full_path)
    ft_model.cuda()
    test_loader = val_loader()

    print("Start testing...")
    test_acc = validate_model(test_loader, ft_model, None)
    print("Testing accuracy is: {:.3f}".format(test_acc))
