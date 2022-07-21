import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
from ltir.utils.loss import CrossEntropy2d
from ltir.model.discriminator import FCDiscriminator

from ltir.dataset.gta5_dataset import GTA5DataSet
from ltir.dataset.cityscapes_dataset import cityscapesDataSet
from ltir.dataset.cityscapes_dataset_label import cityscapesDataSetLabel
from ltir.dataset.cityscapes_dataset_SSL import cityscapesDataSetSSL
from ltir.domain_adaptation.eval_UDA import eval_single
from pathlib import Path
import sys

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def loss_calc(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(cfg, optimizer, i_iter):
    lr = lr_poly(cfg.TRAIN.LEARNING_RATE, i_iter, cfg.TRAIN.NUM_STEPS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(cfg, optimizer, i_iter):
    lr = lr_poly(cfg.TRAIN.LEARNING_RATE, i_iter, cfg.TRAIN.NUM_STEPS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def train_UDA_DeepLab_MobileNet(model, cfg):
    """Create the model and start the training."""
    if os.path.exists(Path(cfg.TRAIN.SNAPSHOT_DIR) / f'output.txt'):
        os.remove(Path(cfg.TRAIN.SNAPSHOT_DIR) / f'output.txt')

    w, h = map(int, cfg.INPUT_SIZE_SOURCE.split(','))
    input_size_source = (w, h)

    w, h = map(int, cfg.INPUT_SIZE_TARGET.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    model.train()
    model.to(cfg.DEVICE)
    cudnn.benchmark = True
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE), lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer.zero_grad()

    ###### DISCRIMINATOR
    model_D = FCDiscriminator(num_classes=cfg.NUM_CLASSES)
    model_D.train()
    model_D.to(cfg.DEVICE)

    optimizer_D = optim.Adam(model_D.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    source_adv_label = 0
    target_adv_label = 1

    ###### DATALOADERS

    trainloader = data.DataLoader(
        GTA5DataSet(cfg.TRANSLATED_DATA_DIR, cfg.DATA_SRC_DIRECTORY, cfg.DATA_LIST_SOURCE,
                    max_iters=cfg.TRAIN.NUM_STEPS * cfg.TRAIN.ITER_SIZE * cfg.TRAIN.BATCH_SIZE,
                    crop_size=input_size_source, scale=cfg.TRAIN.SCALE, mirror=cfg.TRAIN.MIRROR, mean=IMG_MEAN),
        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    style_trainloader = data.DataLoader(
        GTA5DataSet(cfg.STYLIZED_DATA_DIR, cfg.DATA_SRC_DIRECTORY, cfg.DATA_LIST_SOURCE,
                    max_iters=cfg.TRAIN.NUM_STEPS * cfg.TRAIN.ITER_SIZE * cfg.TRAIN.BATCH_SIZE,
                    crop_size=input_size_source, scale=cfg.TRAIN.SCALE, mirror=cfg.TRAIN.MIRROR, mean=IMG_MEAN),
        batch_size=cfg.TRAIN.SCALE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    style_trainloader_iter = enumerate(style_trainloader)

    if cfg.STAGE == 1:
        targetloader = data.DataLoader(cityscapesDataSet(cfg.DATA_TGT_DIRECTORY, cfg.DATA_LIST_TARGET,
                                                         max_iters=cfg.TRAIN.NUM_STEPS * cfg.TRAIN.ITER_SIZE * cfg.TRAIN.BATCH_SIZE,
                                                         crop_size=input_size_target,
                                                         mean=IMG_MEAN,
                                                         set="train"),
                                       batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True)

        targetloader_iter = enumerate(targetloader)
    else:
        # Dataloader for self-training
        targetloader = data.DataLoader(cityscapesDataSetSSL(cfg.DATA_TGT_DIRECTORY, cfg.DATA_LIST_TARGET,
                                                            max_iters=cfg.TRAIN.NUM_STEPS * cfg.TRAIN.ITER_SIZE * cfg.TRAIN.BATCH_SIZE,
                                                            crop_size=input_size_target,
                                                            mean=IMG_MEAN,
                                                            set="train",
                                                            label_folder='Path to generated pseudo labels'),
                                       batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True)

        targetloader_iter = enumerate(targetloader)

    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    for i_iter in range(0, cfg.TRAIN.NUM_STEPS):
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        # train segementation network
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # train with source
        if cfg.STAGE == 1:
            if i_iter % 2 == 0:
                _, batch = next(trainloader_iter)
            else:
                _, batch = next(style_trainloader_iter)

        else:
            _, batch = next(trainloader_iter)

        ##### TRAIN WITH SOURCE
        image_source, label, _, _ = batch
        image_source = image_source.to(cfg.DEVICE)

        pred_source = model(image_source, big_model=True)
        pred_source = interp(pred_source)

        loss_seg_source = loss_calc(pred_source, label, cfg.DEVICE)
        loss_seg_source.backward()

        if cfg.STAGE == 2:
            ##### TRAIN DEEPLAB WITH TARGET IN STAGE 2: SSL
            _, batch = next(targetloader_iter)
            image_target, target_label, _, _ = batch
            image_target = image_target.to(cfg.DEVICE)

            pred_target = model(image_target, big_model=True)
            pred_target = interp_target(pred_target)

            # target segmentation loss
            loss_seg_target = loss_calc(pred_target, target_label, gpu=cfg.DEVICE)
            loss_seg_target.backward()

            pred_target_mobile = model(image_target, big_model=False)
            pred_target_mobile = interp_target(pred_target_mobile)
            loss_seg_target_mobile = loss_calc(pred_target_mobile, target_label.clone(), gpu=cfg.DEVICE)
            loss_seg_target_mobile.backward()

        # optimize
        optimizer.step()

        if cfg.STAGE == 1:
            ##### TRAIN WITH TARGET IN STAGE 1
            _, batch = next(targetloader_iter)
            image_target, _, _ = batch
            image_target = image_target.to(cfg.DEVICE)

            pred_target = model(image_target)
            pred_target = interp_target(pred_target)

            # output-level adversarial training
            D_output_target = model_D(F.softmax(pred_target))
            loss_adv = bce_loss(D_output_target,
                                Variable(torch.FloatTensor(D_output_target.data.size()).fill_(source_adv_label)).to(
                                    cfg.DEVICE))
            loss_adv = loss_adv * cfg.TRAIN.LAMBDA_ADV
            loss_adv.backward()

            # train discriminator
            for param in model_D.parameters():
                param.requires_grad = True

            pred_source = pred_source.detach()
            pred_target = pred_target.detach()

            D_output_source = model_D(F.softmax(pred_source))
            D_output_target = model_D(F.softmax(pred_target))

            loss_D_source = bce_loss(D_output_source, Variable(
                torch.FloatTensor(D_output_source.data.size()).fill_(source_adv_label)).to(cfg.DEVICE))
            loss_D_target = bce_loss(D_output_target, Variable(
                torch.FloatTensor(D_output_target.data.size()).fill_(target_adv_label)).to(cfg.DEVICE))

            loss_D_source = loss_D_source / 2
            loss_D_target = loss_D_target / 2

            loss_D_source.backward()
            loss_D_target.backward()

            optimizer_D.step()

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')

        if i_iter % cfg.TRAIN.VALIDATE == 0 and i_iter != 0:
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            file1 = open(snapshot_dir / f'output.txt', "a")
            file1.write("\n ITERATION : " + str(i_iter) + "\n")
            eval_single(cfg, model, True, file1)
            model.train()
            file1.close()
        elif i_iter % 20 == 0:
            print("Iteration: ", i_iter)
        sys.stdout.flush()




