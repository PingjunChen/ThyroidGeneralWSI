# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
import time, json
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from wsinet import WsiNet
from thyroid_dataset import ThyroidDataSet


def load_wsinet(args):
    wsinet = WsiNet(class_num=args.class_num, in_channels=args.input_fea_num, mode=args.mode)
    weightspath = os.path.join(args.data_dir, "Models/SlideModels/BestModels", args.model_type, args.mode, args.wsi_cls_name)
    # weightspath = os.path.join(args.data_dir, "Models/SlideModels", args.model_type, args.mode, args.wsi_cls_name)
    wsi_weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    wsinet.load_state_dict(wsi_weights_dict)
    wsinet.cuda()
    wsinet.eval()

    return wsinet


def test_cls(net, dataloader):
    start_timer = time.time()
    total_pred, total_pred_probs, total_gt = [], [], []
    for ind, (batch_feas, gt_classes, true_num, bboxes) in enumerate(dataloader):
        # print("Pred {:03d}/{:03d}".format(ind+1, len(dataloader)))
        im_data = Variable(batch_feas.cuda())
        cls_probs, assignments = net(im_data, None, true_num=true_num)
        _, cls_labels = torch.topk(cls_probs.cpu(), 1, dim=1)
        cls_labels = cls_labels.numpy()[:, 0]
        total_gt.extend(gt_classes.tolist())
        total_pred_probs.extend(cls_probs.tolist())
        total_pred.extend(cls_labels.tolist())

    con_mat = confusion_matrix(total_gt, total_pred)
    cur_eval_acc = np.trace(con_mat) * 1.0 / np.sum(con_mat)

    total_time = time.time()-start_timer
    print("Testing Acc: {:.3f}".format(cur_eval_acc))
    # print("Confusion matrix:")
    # print(con_mat)
    benign_pre, benign_recall = con_mat[0,0]/np.sum(con_mat[:, 0]), con_mat[0,0]/np.sum(con_mat[0, :])
    print("   Benign: Precision {:.3f} Recall {:.3f}".format(benign_pre, benign_recall))
    uncertain_pre, uncertain_recall = con_mat[1,1]/np.sum(con_mat[:, 1]), con_mat[1,1]/np.sum(con_mat[1, :])
    print("Uncertain: Precision {:.3f} Recall {:.3f}".format(uncertain_pre, uncertain_recall))
    malignant_pre, malignant_recall = con_mat[2,2]/np.sum(con_mat[:, 2]), con_mat[2,2]/np.sum(con_mat[2, :])
    print("Malignant: Precision {:.3f} Recall {:.3f}".format(malignant_pre, malignant_recall))

    test_gt_pred = {}
    save_json = False
    if save_json == True:
        test_gt_pred['preds'] = total_pred_probs
        test_gt_pred['gts'] = to_categorical(total_gt).tolist()
        json_path = os.path.join("../Vis", "pred_gt.json")
        with open(json_path, 'w') as fp:
            json.dump(test_gt_pred, fp)


def set_args():
    parser = argparse.ArgumentParser(description = 'Thyroid WSI diagnois')
    parser.add_argument("--batch_size",      type=int,   default=24,      help="batch size")
    parser.add_argument('--device_id',       type=str,   default="2",     help='which device')
    parser.add_argument('--test_num',        type=int,   default=128,    help='which device')
    # model setting
    parser.add_argument("--class_num",       type=int,   default=3)
    parser.add_argument("--input_fea_num",   type=int,   default=4096)
    parser.add_argument('--model_type',      type=str,   default="vgg16bn")
    parser.add_argument("--data_dir",        type=str,   default="../data/CV04")
    parser.add_argument("--mode",            type=str,   default="selfatt")
    parser.add_argument('--wsi_cls_name',    type=str,   default="epoch-100-acc-0.779.pth")
    parser.add_argument("--pre_load",        action='store_true', default=True)
    parser.add_argument('--verbose',         action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    # model
    wsi_net = load_wsinet(args)
    # dataset
    test_data_root = os.path.join(args.data_dir, "Feas", args.model_type, "test")
    test_dataset = ThyroidDataSet(test_data_root, testing=True, testing_num=args.test_num, pre_load=args.pre_load)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size= args.batch_size, num_workers=4, pin_memory=True)

    print(">> START testing {} with {}".format(args.model_type, args.mode))
    print(">> model name: {} with test_num: {}".format(args.wsi_cls_name, args.test_num))
    test_cls(wsi_net, test_dataloader)
