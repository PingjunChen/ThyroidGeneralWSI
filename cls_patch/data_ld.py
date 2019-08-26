# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from scipy import misc

import torch.utils.data
from torchvision import datasets, transforms


ClassNUM = 3
DataRoot = "../data/Patches/"
TrainDir = os.path.join(DataRoot, 'Train')
ValDir = os.path.join(DataRoot, 'Val')


def find_ext_files(dir_name, ext):
    assert os.path.isdir(dir_name), "{} is not a valid directory".format(dir_name)

    file_list = []
    for root, _, files in os.walk(dir_name):
        for cur_file in files:
            if cur_file.endswith(ext):
                 file_list.append(os.path.join(root, cur_file))

    return file_list

def get_mean_and_std(img_dir, suffix):
    mean, std = np.zeros(3), np.zeros(3)
    filelist = find_ext_files(img_dir, suffix)

    for idx, filepath in enumerate(filelist):
        cur_img = misc.imread(filepath) / 255.0
        for i in range(3):
            mean[i] += cur_img[:,:,i].mean()
            std[i] += cur_img[:,:,i].std()
    mean = [ele * 1.0 / len(filelist) for ele in mean]
    std = [ele * 1.0 / len(filelist) for ele in std]
    return mean, std

# rgb_mean, rgb_std = get_mean_and_std(TrainDir, suffix=".png")
# print("mean rgb: {}".format(rgb_mean))
# print("std rgb: {}".format(rgb_std))

rgb_mean, rgb_std = (0.8071, 0.6344, 0.8158), (0.1319, 0.1650, 0.094)
batch_size = 32


def train_loader():
    kwargs = {"num_workers": 4, "pin_memory": True}
    train_dataset = datasets.ImageFolder(TrainDir,
        transform = transforms.Compose([
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)])
        )

    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    return loader


def val_loader():
    kwargs = {"num_workers": 4, "pin_memory": True}

    val_dataset = datasets.ImageFolder(ValDir,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)])
        )

    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return loader
