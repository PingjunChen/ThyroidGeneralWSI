import os, sys
import numpy as np
import shutil
from random import shuffle
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

import torch
from torch.multiprocessing import Pool
from torch.autograd import Variable
from torch.optim import lr_scheduler


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def train_cls(dataloader, val_dataloader, model_root, net, args):
    net.train()

    start_epoch = 1
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                nesterov=True, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=LambdaLR(args.maxepoch, start_epoch, args.decay_epoch).step)
    train_loss, step_cnt, batch_count = 0.0, 0, 0
    best_acc = 0.0
    for epoc_num in np.arange(start_epoch, args.maxepoch+1):
        for batch_idx, (batch_data, gt_classes, true_num, bboxes) in enumerate(dataloader):
            im_data   = batch_data.cuda().float()
            im_label  = gt_classes.cuda().long()
            num_data  = true_num.cuda().long()

            im_label = im_label.view(-1, 1)
            train_pred, assignments = net(im_data, im_label, true_num=num_data)

            vecloss = net.loss
            loss = torch.mean(vecloss)
            n_data = im_data.size()[0]
            num_sample  = im_data.size()[0]
            train_loss_val = loss.data.cpu().item()
            train_loss  += train_loss_val
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_cnt += 1
            batch_count += 1

            train_loss /= step_cnt
        print((' epoch {}, loss: {}, learning rate: {:.5f}'. \
                format(epoc_num, train_loss, optimizer.param_groups[0]['lr'])))

        net.eval()
        total_pred, total_gt = [], []
        for val_data, val_label, val_num, val_boxes in val_dataloader:
            val_data  =  val_data.cuda().float()
            val_num   =  val_num.cuda().long()
            val_pred_pro, assignments = net(val_data, true_num = val_num)
            val_pred_pro = val_pred_pro.cpu()
            _, cls_labels  = torch.topk(val_pred_pro, 1, dim=1)
            cls_labels    =  cls_labels.data.cpu().numpy()[:,0]

            total_pred.extend(cls_labels.tolist())
            total_gt.extend(val_label.tolist())
        precision, recall, fscore, support = score(total_gt, total_pred)
        con_mat = confusion_matrix(total_gt, total_pred)
        # print(' p:  {}\n r:  {}\n f1: {} \n'.format(precision, recall, fscore))
        # print('confusion matrix:')
        # print(con_mat)
        cls_acc = np.trace(con_mat) * 1.0 / np.sum(con_mat)
        print("\n Current classification accuracy is: {:.4f}".format(cls_acc))
        train_loss, step_cnt = 0, 0
        net.train()

        lr_scheduler.step()
        if epoc_num % args.save_freq == 0 and cls_acc >= best_acc:
            save_model_name = 'epoch-{}-acc-{:.3f}.pth'.format(str(epoc_num).zfill(3), cls_acc)
            torch.save(net.state_dict(), os.path.join(model_root, save_model_name))
            print('Model saved as {}'.format(save_model_name))
            best_acc = cls_acc
