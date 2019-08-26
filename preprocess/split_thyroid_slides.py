# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import shutil
from pydaily import format

label_dict = {
    "Benign": 1,
    "Uncertain": 2,
    "Malignant": 3,
}


if __name__ == "__main__":
    np.random.seed(1234)

    split_ratio = 0.2
    thyroid_data_dir = "/media/pingjun/Pingjun350/ThyroidData"
    benign_list, uncertain_list, maglignant_list = [], [], []
    # get label from annotations
    annotation_dir = os.path.join(thyroid_data_dir, "Training", "Annotations")
    slide_list = [ele for ele in os.listdir(annotation_dir) if "json" in ele]
    for ele in slide_list:
        cur_slide_json = os.path.join(annotation_dir, ele)
        annotation_dict = format.json_to_dict(cur_slide_json)
        slide_id = annotation_dict['img_name']
        slide_diagnosis = annotation_dict['conclusion']
        if slide_diagnosis == "Benign":
            benign_list.append(slide_id)
        elif slide_diagnosis == "Uncertain":
            uncertain_list.append(slide_id)
        elif slide_diagnosis == "Malignant":
            maglignant_list.append(slide_id)
        else:
            print("Unknow slides")
    print("There are {} benign, {} uncertain, and {} malignant slides.".format(
        len(benign_list), len(uncertain_list), len(maglignant_list)))
    train_benign_num = int((1 - split_ratio) * len(benign_list))
    train_uncertain_num = int((1 - split_ratio) * len(uncertain_list))
    train_malignant_num = int((1 - split_ratio) * len(maglignant_list))

    train_benign_list = np.random.choice(benign_list, train_benign_num, replace=False)
    val_benign_list = list(set(benign_list) - set(train_benign_list))
    train_uncertain_list = np.random.choice(uncertain_list, train_uncertain_num, replace=False)
    val_uncertain_list = list(set(uncertain_list) - set(train_uncertain_list))
    train_malignant_list = np.random.choice(maglignant_list, train_malignant_num, replace=False)
    val_malignant_list = list(set(maglignant_list) - set(train_malignant_list))

    # copy to train
    print("Copy Train Benign")
    for ele in train_benign_list:
        src_annotation_path = os.path.join(thyroid_data_dir, "Training", "Annotations", ele+".json")
        dst_annotation_path = os.path.join(thyroid_data_dir, "TrainVal", "Annotations/Train/1Benign")
        shutil.copy(src_annotation_path, dst_annotation_path)
        src_slide_path = os.path.join(thyroid_data_dir, "Training", "Slides", ele+".tiff")
        dst_slide_path = os.path.join(thyroid_data_dir, "TrainVal", "Slides/Train/1Benign")
        shutil.copy(src_slide_path, dst_slide_path)
    print("Copy Train Uncertain")
    for ele in train_uncertain_list:
        src_annotation_path = os.path.join(thyroid_data_dir, "Training", "Annotations", ele+".json")
        dst_annotation_path = os.path.join(thyroid_data_dir, "TrainVal", "Annotations/Train/2Uncertain")
        shutil.copy(src_annotation_path, dst_annotation_path)
        src_slide_path = os.path.join(thyroid_data_dir, "Training", "Slides", ele+".tiff")
        dst_slide_path = os.path.join(thyroid_data_dir, "TrainVal", "Slides/Train/2Uncertain")
        shutil.copy(src_slide_path, dst_slide_path)
    print("Copy Train Malignant")
    for ele in train_malignant_list:
        src_annotation_path = os.path.join(thyroid_data_dir, "Training", "Annotations", ele+".json")
        dst_annotation_path = os.path.join(thyroid_data_dir, "TrainVal", "Annotations/Train/3Malignant")
        shutil.copy(src_annotation_path, dst_annotation_path)
        src_slide_path = os.path.join(thyroid_data_dir, "Training", "Slides", ele+".tiff")
        dst_slide_path = os.path.join(thyroid_data_dir, "TrainVal", "Slides/Train/3Malignant")
        shutil.copy(src_slide_path, dst_slide_path)

    # copy to val
    print("Copy Val Benign")
    for ele in val_benign_list:
        src_annotation_path = os.path.join(thyroid_data_dir, "Training", "Annotations", ele+".json")
        dst_annotation_path = os.path.join(thyroid_data_dir, "TrainVal", "Annotations/Val/1Benign")
        shutil.copy(src_annotation_path, dst_annotation_path)
        src_slide_path = os.path.join(thyroid_data_dir, "Training", "Slides", ele+".tiff")
        dst_slide_path = os.path.join(thyroid_data_dir, "TrainVal", "Slides/Val/1Benign")
        shutil.copy(src_slide_path, dst_slide_path)
    print("Copy Val Uncertain")
    for ele in val_uncertain_list:
        src_annotation_path = os.path.join(thyroid_data_dir, "Training", "Annotations", ele+".json")
        dst_annotation_path = os.path.join(thyroid_data_dir, "TrainVal", "Annotations/Val/2Uncertain")
        shutil.copy(src_annotation_path, dst_annotation_path)
        src_slide_path = os.path.join(thyroid_data_dir, "Training", "Slides", ele+".tiff")
        dst_slide_path = os.path.join(thyroid_data_dir, "TrainVal", "Slides/Val/2Uncertain")
        shutil.copy(src_slide_path, dst_slide_path)
    print("Copy Val Malignant")
    for ele in val_malignant_list:
        src_annotation_path = os.path.join(thyroid_data_dir, "Training", "Annotations", ele+".json")
        dst_annotation_path = os.path.join(thyroid_data_dir, "TrainVal", "Annotations/Val/3Malignant")
        shutil.copy(src_annotation_path, dst_annotation_path)
        src_slide_path = os.path.join(thyroid_data_dir, "Training", "Slides", ele+".tiff")
        dst_slide_path = os.path.join(thyroid_data_dir, "TrainVal", "Slides/Val/3Malignant")
        shutil.copy(src_slide_path, dst_slide_path)
