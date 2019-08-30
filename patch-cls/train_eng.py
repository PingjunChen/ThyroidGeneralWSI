# -*- coding: utf-8 -*-

import os, sys
import shutil

from torchvision import models
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from sklearn import metrics
import numpy as np
from patchloader import train_loader, val_loader


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (args.lr_decay_ratio ** (epoch // args.lr_decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_thyroid(args):
    # construct model
    if args.model_name == "inceptionv3":
        model = models.inception_v3(pretrained=True)
        model.fc = nn.Linear(2048, args.class_num)
    elif args.model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(512*4, args.class_num)
    elif args.model_name == "vgg16bn":
        model = models.vgg16_bn(pretrained=True)
        model.classifier[-1] = nn.Linear(4096, args.class_num)
    else:
        raise AssertionError("unknown model name")


    model.cuda()
    # optimizer & loss
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          weight_decay=5.0e-4, momentum=0.9, nesterov=True)
    criterion =nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 1.8, 1.5]).cuda())

    # dataloader
    train_thyroid_loader = train_loader(args.batch_size, args.patch_size)
    val_thyroid_loader = val_loader(args.batch_size, args.patch_size)

    # folder for model saving
    model_dir = os.path.join(args.model_dir, args.model_name, args.session)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    best_acc = 0.0
    for epoch in range(0, args.epochs):
        print("Current learning rate is: {:.6f}".format(optimizer.param_groups[0]['lr']))
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train_model(train_thyroid_loader, model, criterion, optimizer, epoch, args)
        # evaluate on validation set
        val_acc = validate_model(val_thyroid_loader, model, args)
        is_best = val_acc > best_acc
        if is_best == True:
            best_acc = val_acc
            cur_model_name = args.data_name + str(epoch).zfill(2) + "-{:.4f}.pth".format(best_acc)
            torch.save(model.cpu(), os.path.join(model_dir, cur_model_name))
            print('Save weights at {}/{}'.format(model_dir, cur_model_name))
            model.cuda()


def train_model(train_loader, model, criterion, optimizer, epoch, args):
    model.train()

    correct, total, ttl_loss = 0.0, 0.0, 0.0
    gt_labels, pred_labels = [], []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        gt_labels.extend(targets.tolist())
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        if args.model_name == "inceptionv3":
            outputs, _ = model(inputs)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        ttl_loss += loss.item()

        _, prediction = outputs.max(1)
        pred_labels.extend(prediction.cpu().tolist())
        total += targets.size(0)
        correct += prediction.eq(targets).sum().item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            batch_progress = 100. * batch_idx / len(train_loader)
            print("Train Epoch: {} [{}/{} ({:.2f})%)]".format(
                epoch, batch_idx, len(train_loader), batch_progress))
    print("Training accuracy on Epoch {} is [{}/{} {:.4f})]\t Loss: {:.4f}".format(
        epoch, correct, total, correct/total, ttl_loss/total))
    train_cm = metrics.confusion_matrix(gt_labels, pred_labels)
    print("Training confusion matrix:")
    print(train_cm)


def validate_model(val_loader, model, args):
    model.eval()
    gt_labels, pred_labels = [], []
    with torch.no_grad():
        correct, total = 0, 0
        for i, (inputs, targets) in enumerate(val_loader):
            gt_labels.extend(targets.tolist())
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)

            _, prediction = outputs.max(1)
            pred_labels.extend(prediction.cpu().tolist())
            total += targets.size(0)
            correct += prediction.eq(targets).sum().item()
        val_acc = correct * 1.0 / total
        print("Validation accuracy is [{}/{} {:.4f})]".format(correct, total, val_acc))
    val_cm = metrics.confusion_matrix(gt_labels, pred_labels)
    print("Validatioin confusion matrix:")
    print(val_cm)

    return val_acc
