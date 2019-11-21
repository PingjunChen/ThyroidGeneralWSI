# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import deepdish as dd
import math, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from thyroid_config import multi_class_map_dict
from thyroid_config import folder_map_dict, folder_reverse_map
from utils import aggregate_label, get_all_files


class ThyroidDataSet(Dataset):
    def __init__(self, data_dir, testing=False, testing_num=128, pre_load=True):
        self.data_dir = data_dir
        self.pre_load = pre_load
        self.testing = testing
        self.testing_num = testing_num

        # category configuration
        self.folder_map_dict = folder_map_dict
        self.folder_reverse_map = folder_reverse_map
        self.class_map_dict = multi_class_map_dict
        self.cur_filename = None

        file_list, label_list = get_all_files(data_dir, inputext = ['.h5'],
                                              class_map_dict=self.folder_map_dict,
                                              pre_load=self.pre_load)
        self.file_list   = file_list
        self.label_list  = label_list

        summery_label_dict = aggregate_label(label_list)
        self.label_dict   =  summery_label_dict
        self.img_num      =  len(self.file_list)
        self.indices = list(range(self.img_num))

        self.chosen_num_list = list(range(testing_num, testing_num+64))
        self.max_num = testing_num+64 if not self.testing else self.testing_num
        self.fixed_num = 40
        self.additoinal_num = 20


    def get_true_label(self, label):
        new_label =  self.class_map_dict[self.folder_reverse_map[label]]
        return new_label

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        if self.pre_load == True:
            data = self.file_list[index]
        else:
            this_data_path = self.file_list[index]
            data = dd.io.load(this_data_path)
            self.cur_filename = os.path.basename(this_data_path)

        probs, feas, bboxes = data['probs'], data['feas'], data['bboxes']
        total_ind  = np.array(range(0, len(probs)))
        feas_placeholder = np.zeros((self.max_num, feas.shape[1]), dtype=np.float32)
        box_placeholder = np.zeros((self.max_num, bboxes.shape[1]), dtype=np.int64)

        if self.testing:
            if len(probs) > self.testing_num:
                chosen_total_ind_ = total_ind[:self.testing_num]
            else:
                chosen_total_ind_ = total_ind
        else:
            if len(probs) > self.max_num:
                chosen_num = random.choice(self.chosen_num_list)
                # front fixed chosen part
                fixed_chosen_num = self.fixed_num+self.additoinal_num
                fixed_chosen_ind = np.random.choice(total_ind[:fixed_chosen_num], self.fixed_num)
                # later random chosen part
                random_chosen_num = chosen_num-self.fixed_num
                pos_probs = np.sum(probs[fixed_chosen_num:, 1:], axis=1)
                random_chosen_probs = pos_probs / np.sum(pos_probs)
                random_chosen_ind = np.random.choice(total_ind[fixed_chosen_num:], random_chosen_num, replace=False, p=random_chosen_probs)
                # merge indices
                chosen_total_ind_ = np.concatenate([fixed_chosen_ind, random_chosen_ind], 0)
            else:
                chosen_total_ind_ = total_ind

        # extract chosen patches
        true_num = len(chosen_total_ind_)
        feas_placeholder[:true_num] = feas[chosen_total_ind_]
        this_true_label = self.get_true_label(self.label_list[index])
        box_placeholder[:true_num] = bboxes[chosen_total_ind_]

        return feas_placeholder, this_true_label, true_num, box_placeholder
