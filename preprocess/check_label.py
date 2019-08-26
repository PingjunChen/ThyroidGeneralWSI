# -*- coding: utf-8 -*-

import os, sys
from pydaily import format

label_dict = {
    "Benign": 1,
    "Uncertain": 2,
    "Malignant": 3,
}

if __name__ == "__main__":
    thyroid_data_dir = "/media/pingjun/Pingjun350/ThyroidData"
    # Previous splitting
    train_slides_json = os.path.join(thyroid_data_dir, "Diagnosis", "train_slides_gt.json")
    val_slides_json = os.path.join(thyroid_data_dir, "Diagnosis", "val_slides_gt.json")
    train_slides_dict = format.json_to_dict(train_slides_json)
    val_slides_dict = format.json_to_dict(val_slides_json)
    train_slides_dict.update(val_slides_dict)

    # get label from annotations
    annotation_dir = os.path.join(thyroid_data_dir, "Training", "Annotations")
    slide_list = [ele for ele in os.listdir(annotation_dir) if "json" in ele]
    assert len(slide_list) == len(train_slides_dict), "slide number not matching"
    for ele in slide_list:
        cur_slide_json = os.path.join(annotation_dir, ele)
        annotation_dict = format.json_to_dict(cur_slide_json)
        slide_id = annotation_dict['img_name']
        slide_label = label_dict[annotation_dict['conclusion']]
        if train_slides_dict[slide_id] != slide_label:
            print("{} annotation not matching".format(ele))
