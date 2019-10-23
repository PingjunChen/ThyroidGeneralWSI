# -*- coding: utf-8 -*-

import os, sys
import argparse
import torch
import torch.backends.cudnn as cudnn

from train_eng import validate_model
from patchloader import val_loader


def set_args():
    parser = argparse.ArgumentParser(description='Thyroid Classification')
    parser.add_argument('--seed',            type=int,   default=1234)
    parser.add_argument('--model_dir',       type=str,   default="../data/CV03/Models/PatchModels")
    parser.add_argument('--data_name',       type=str,   default="thyroid")
    parser.add_argument('--model_name',      type=str,   default="vgg16bn")
    parser.add_argument('--model_path',      type=str,   default="thyroid00-0.7721.pth")
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--patch_size',      type=int,   default=224)
    parser.add_argument('--device_id',       type=str,   default="3",     help='which device')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    model_full_path = os.path.join(args.model_dir, args.model_name, args.model_path)
    ft_model = torch.load(model_full_path)
    ft_model.cuda()
    test_loader = val_loader(args.batch_size, args.patch_size)

    print("Start testing...")
    print("Mode: {}, name: {}".format(args.model_name, args.model_path))
    test_acc = validate_model(test_loader, ft_model, None)
    print("Testing accuracy is: {:.3f}".format(test_acc))
