# -*- coding: utf-8 -*-

import os, sys

folder_ratio_map = {
    "1Benign": 0.4,
    "2Uncertain": 0.3,
    "3Malignant": 0.3
}


multi_class_map_dict = {
    "1Benign": 0,
    "2Uncertain": 1,
    "3Malignant": 2,
}


"""
class_reverse_map = {
    0: "1Benign",
    1: "2Uncertain",
    2: "3Malignant",
}
"""
class_reverse_map = {}
for k, v in multi_class_map_dict.items():
    class_reverse_map[v] = k


"""
folder_map_dict = {
    "1Benign": 0,
    "2Uncertain": 1,
    "3Malignant": 2,
}
"""
folder_map_dict = {}
for idx, (k, v ) in enumerate(folder_ratio_map.items()):
    folder_map_dict[k] = idx


"""
folder_reverse_map = {
    0: "1Benign",
    1: "2Uncertain",
    2: "3Malignant",
}
"""
folder_reverse_map = {}
for k, v in folder_map_dict.items():
    folder_reverse_map[v] = k
