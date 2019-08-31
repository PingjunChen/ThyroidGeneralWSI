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
from thyroid_config import folder_map_dict, folder_reverse_map, folder_ratio_map
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
        self.folder_ratio_map = folder_ratio_map
        self.class_map_dict = multi_class_map_dict
        self.cur_filename = None

        # calculate class_ratio_array
        class_ratio_array = [None]*len(self.folder_map_dict.keys())
        for this_k in self.folder_map_dict.keys():
            class_ratio_array[self.folder_map_dict[this_k]] = self.folder_ratio_map[this_k]

        class_ratio_array = np.asarray(class_ratio_array).astype(np.float)
        class_ratio_array = class_ratio_array/np.sum(class_ratio_array)
        self.class_ratio_array = class_ratio_array

        file_list, label_list = get_all_files(data_dir, inputext = ['.h5'],
                                              class_map_dict=self.folder_map_dict, pre_load=self.pre_load)
        self.file_list   = file_list
        self.label_list  = label_list

        summery_label_dict = aggregate_label(label_list)
        self.label_dict   =  summery_label_dict
        self.img_num      =  len(self.file_list)
        self.indices = list(range(self.img_num))

        self.chosen_num_list = list(range(testing_num, testing_num+64))
        self.max_num = testing_num+64 if not self.testing else self.testing_num
        self.fixed_num = 40
        self.additoinal_num = 10


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
        feas = np.squeeze(feas)
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
                random_chosen_ind = np.random.choice(total_ind[fixed_chosen_num:], random_chosen_num,
                                                     replace=False, p=random_chosen_probs)
                chosen_total_ind_ = np.concatenate([fixed_chosen_ind, random_chosen_ind], 0)
            else:
                chosen_total_ind_ = total_ind

        chosen_total_ind_ = chosen_total_ind_.reshape((chosen_total_ind_.shape[0],))
        chosen_feas = feas[chosen_total_ind_]
        true_num = chosen_feas.shape[0]
        box_placeholder[0:true_num] = bboxes[chosen_total_ind_]
        feas_placeholder[0:true_num] = chosen_feas
        this_true_label = self.get_true_label(self.label_list[index])

        return feas_placeholder, this_true_label, true_num, box_placeholder



class BatchSampler(object):
    def __init__(self, label_dict=None, batch_size=24,
                 class_ratio_array=None, num_sampling=8, data_len=None):
        self.label_dict = label_dict
        self.batch_size = batch_size
        self.num_sampling = num_sampling
        self.class_ratio_array = class_ratio_array
        self.data_len = data_len
        self.num_batches = self.data_len // self.batch_size


    def get_indices_balance(self, label_dict, num_sampling, batch_size, class_ratio_array):
        '''
        Parameters:
        -----------
            label_dict:   a dictionary of cls:idx
            num_sampling: the number of chosen class in each run
            batch_size:   totoal number of samples in each run
            class_ratio_array: class_ratio_array[8] store prob of class_reverse_label[8]
        Return:
        -----------
            indices of the sample id
        '''

        indices = []
        key_list = list(label_dict.keys())
        prob = class_ratio_array.copy()[0:len(key_list)]

        for idx, this_k in enumerate(key_list):
            prob[idx] = class_ratio_array[this_k]
        prob = prob/np.sum(prob)

        true_sampling = min(num_sampling, len(key_list))
        each_chosen = math.ceil(batch_size/true_sampling)

        chosen_key = np.random.choice(key_list, true_sampling, replace=False, p=prob)
        for idx, this_key in enumerate(chosen_key):
            #this_key = chosen_key[idx]
            this_ind = label_dict[this_key] #idx set of all the img in this classes.
            this_num = min(each_chosen,  batch_size - each_chosen*idx)

            if this_num <= len(this_ind):
                this_choice = np.random.choice(this_ind, this_num, replace=False)
            else:
                this_choice = np.random.choice(this_ind, this_num, replace=True)

            indices.extend(this_choice)

        return indices

    def __iter__(self):
        for idx in range( self.num_batches):
            batch = self.get_indices_balance(self.label_dict,
                self.num_sampling, self.batch_size, self.class_ratio_array)
            yield batch

    def __len__(self):
        return self.num_batches
