# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

import openslide
from skimage import color, filters
from skimage import img_as_ubyte
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from shapely.geometry import Polygon
import cv2
import math


class PatchDataset(data.Dataset):
    """
    Dataset for thyroid slide testing. Thyroid slides would be splitted into multiple patches.
    Prediction is made on these splitted patches.
    """

    def __init__(self, slide_patches, patch_size):
        self.patches = slide_patches
        self.rgb_mean = (0.800, 0.630, 0.815)
        self.rgb_std = (0.136, 0.168, 0.096)
        self.transform = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize(patch_size), transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        sample = self.patches[idx,...]
        if self.transform:
            sample = self.transform(sample)

        return sample


def extract_deep_feas(model, inputs, model_name):
    model.eval()
    if "inceptionv3" == model_name:
        if model.transform_input:
            x_ch0 = torch.unsqueeze(inputs[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(inputs[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(inputs[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = model.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        fea = x.view(x.size(0), -1)
        # N x 2048
        logit = model.fc(fea)
        prob = F.softmax(logit, dim=-1)
    elif "vgg16bn" == model_name:
        x = model.features(inputs)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        fea = model.classifier[:4](x)
        logit = model.classifier[4:](fea)
        prob = F.softmax(logit, dim=-1)
    elif "resnet50" == model_name:
        x = model.conv1(inputs)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        fea = x.reshape(x.size(0), -1)
        logit = model.fc(fea)
        prob = F.softmax(logit, dim=-1)
    else:
        raise AssertionError("Unknown model name {}".format(model_name))

    return prob, logit, fea


def pred_feas(model, patches, args):
    probs, logits, vecs = [], [], []
    slide_dset = PatchDataset(patches, args.patch_size)
    dset_loader = data.DataLoader(slide_dset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, drop_last=False)
    with torch.no_grad():
        for ind, inputs in enumerate(dset_loader):
            x = Variable(inputs.cuda())
            prob, logit, fea = extract_deep_feas(model, x, args.model_type)
            probs.extend(prob.cpu().numpy())
            logits.extend(logit.cpu().numpy())
            vecs.extend(fea.cpu().numpy())

    return probs, logits, vecs


def gen_slide_feas(cls_model, split_arr, patches, wsi_dim, args):
    RAW_SIZE = 256
    SIZE1, SIZE2, SIZE4 = int(RAW_SIZE/4), int(RAW_SIZE/2), RAW_SIZE
    class_num = 3

    FeasList = []
    BBoxes, ClsProbs, ClsLogits, FeaVecs = [], [], [], []
    # Prediction
    if patches.shape[0] > 0: # exist
        ClsProbs, ClsLogits, FeaVecs = pred_feas(cls_model, patches, args)
        for coor in split_arr:
            cur_x, cur_y = coor[1]+SIZE1, coor[0]+SIZE1
            cur_bbox = [cur_x, cur_y, SIZE2, SIZE2]
            BBoxes.append(cur_bbox)

    norm_prob_list = [ele[0] for ele in ClsProbs]
    sorting_indx = np.argsort(norm_prob_list)

    probs = np.array([ClsProbs[ind] for ind in sorting_indx])
    logits = np.array([ClsLogits[ind] for ind in sorting_indx])
    feas = np.array([FeaVecs[ind] for ind in sorting_indx])
    bboxes = np.array([BBoxes[ind] for ind in sorting_indx])

    return probs, logits, feas, bboxes


# Locate tissue regions from kfb low level image
def find_tissue_cnt(slide_path, level=3, thresh_val=0.82, min_size=2.0e5):
    slide_head = openslide.OpenSlide(slide_path)
    wsi_img = slide_head.read_region((0, 0), level, slide_head.level_dimensions[level])
    wsi_img = np.array(wsi_img)[:,:,:3]
    # Gray
    gray = color.rgb2gray(wsi_img)
    # Smooth
    smooth = filters.gaussian(gray, sigma=9)
    # Threshold
    binary = smooth < thresh_val
    # Fill holes
    fill = binary_fill_holes(binary)
    # Remove outliers
    mask = remove_small_objects(fill, min_size=min_size, connectivity=8)
    # Find contours
    _, cnts, _ = cv2.findContours(img_as_ubyte(mask),
                                  mode=cv2.RETR_EXTERNAL,
                                  method=cv2.CHAIN_APPROX_NONE)
    return wsi_img, mask, cnts


def split_regions(slide_path, img_level=2, cnt_level=3):
    s_img, mask, cnts = find_tissue_cnt(slide_path, cnt_level)
    img_cnt_ratio = 2**(cnt_level-img_level)
    wsi_dim = [ele*img_cnt_ratio for ele in s_img.shape[:2]]
    slide_head = openslide.OpenSlide(slide_path)
    wsi_img = slide_head.read_region((0, 0), img_level, wsi_dim)
    wsi_img = np.array(wsi_img)[:,:,:3]

    RAW_SIZE = 256
    SIZE1, SIZE2, SIZE4 = int(RAW_SIZE/4), int(RAW_SIZE/2), RAW_SIZE
    split_arr, patch_list = [], []
    for c_ind in range(len(cnts)):
        cur_cnt = cnts[c_ind] * img_cnt_ratio
        cur_cnt = np.squeeze(cur_cnt)
        w_coors = [int(round(ele)) for ele in cur_cnt[:, 0].tolist()]
        h_coors = [int(round(ele)) for ele in cur_cnt[:, 1].tolist()]
        w_min, w_max = min(w_coors), max(w_coors)
        h_min, h_max = min(h_coors), max(h_coors)

        # Width range to crop
        start_w = (w_min - SIZE1) if (w_min - SIZE1) > 0 else 0
        num_w = int(math.ceil((w_max - start_w - SIZE2)/SIZE2))
        # Height range to crop
        start_h = (h_min - SIZE1) if (h_min - SIZE1) > 0 else 0
        num_h = int(math.ceil((h_max - start_h - SIZE2)/SIZE2))

        poly_cnt = Polygon(cur_cnt)
        if poly_cnt.is_valid == False:
            continue
        for iw in range(0, num_w):
            for ih in range(0, num_h):
                # determine current rectangular is inside the contour or not
                cur_coors = [(start_w+iw*SIZE2, start_h+ih*SIZE2), (start_w+iw*SIZE2+SIZE4, start_h+ih*SIZE2),
                             (start_w+iw*SIZE2+SIZE4, start_h+ih*SIZE2+SIZE4), (start_w+iw*SIZE2, start_h+ih*SIZE2+SIZE4)]
                if start_w+iw*SIZE2+SIZE4 >= wsi_img.shape[1] or start_h+ih*SIZE2+SIZE4 > wsi_img.shape[0]:
                    continue
                patch_cnt = Polygon(cur_coors)
                try:
                    inter_flag = poly_cnt.intersects(patch_cnt)
                    if inter_flag == False:
                        continue
                    else:
                        inter_cnt = poly_cnt.intersection(patch_cnt)
                        if inter_cnt.area > patch_cnt.area * 0.5:
                            split_arr.append((start_h+ih*SIZE2, start_w+iw*SIZE2))
                            split_patch = wsi_img[start_h+ih*SIZE2:start_h+ih*SIZE2+SIZE4, start_w+iw*SIZE2:start_w+iw*SIZE2+SIZE4, :]
                            patch_list.append(split_patch)
                except:
                    print("Error in Polygon relationship")
    return split_arr, patch_list, wsi_dim, s_img, mask
