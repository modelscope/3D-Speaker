# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script is designed for training an audio-visual active speaker detection system using the TalkNet model,
and has been adapted from the original repository at https://github.com/TaoRuijie/TalkNet-ASD.
"""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import numpy as np

from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy, average_precision
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.utils.epoch import EpochCounter, EpochLogger


parser = argparse.ArgumentParser(description='Active Speaker Detection Training.')
parser.add_argument('--config', default='', type=str, help='Config file for training.')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not.')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--test', dest='test', action='store_true', help='Only do testing.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')


def train_process(config):
    dist.init_process_group(backend='nccl')
    set_seed(args.seed)
    os.makedirs(config.exp_dir, exist_ok=True)
    logger = get_logger('%s/train.log' % config.exp_dir)
    logger.info(f"Use GPU: {gpu} for training.")

    # dataset
    train_dataset = build('train_dataset', config)
    val_dataset = build('val_dataset', config)

    # train dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    config.train_dataloader['args']['sampler'] = train_sampler
    train_dataloader = build('train_dataloader', config)

    # val dataloader
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    config.val_dataloader['args']['sampler'] = val_sampler
    val_dataloader = build('val_dataloader', config)

    # model
    model = build('model', config)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)

    # optimizer
    config.optimizer['args']['params'] = model.parameters()
    optimizer = build('optimizer', config)

    # loss function
    criterion = build('loss', config)

    # scheduler
    lr_scheduler = build('lr_scheduler', config)

    # others
    epoch_counter = build('epoch_counter', config)
    checkpointer = build('checkpointer', config)

    epoch_logger = EpochLogger(save_file=os.path.join(config.exp_dir, 'train_epoch.log'))

    # resume from a checkpoint
    if args.resume:
        checkpointer.recover_if_possible(device='cuda')

    for epoch in epoch_counter:
        train_sampler.set_epoch(epoch)

        # train one epoch
        train_stats = train(
            train_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            logger,
            config,
        )
        lr_scheduler.step()

        if config.rank == 0:
            # log
            epoch_logger.log_stats(
                stats_meta={"epoch": epoch},
                stats=train_stats,
            )
            # save checkpoint
            if epoch % config.save_epoch_freq == 0:
                checkpointer.save_checkpoint(epoch=epoch)

        if epoch % config.evaluate_epoch_freq == 0:
            val_stats = evaluate(
                val_dataloader,
                model,
                criterion,
                epoch,
                logger,
                config
            )
            if config.rank == 0:
                # log
                epoch_logger.log_stats(
                    stats_meta={"epoch": epoch},
                    stats=val_stats,
                    stage='val',
                )

        dist.barrier()

def train(train_loader, model, criterion, optimizer, epoch, logger, config):
    train_stats = AverageMeters()
    train_stats.add('Time', ':6.3f')
    train_stats.add('Data', ':6.3f')
    train_stats.add('Loss', ':.4e')
    train_stats.add('Acc', ':6.2f')
    train_stats.add('Lr', ':.3e')
    progress = ProgressMeter(
        len(train_loader),
        train_stats,
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    for i, (audioX, visualX, labels) in enumerate(train_loader):
        train_stats.update('Data', time.time() - end)

        audioX = audioX.flatten(start_dim=0, end_dim=1).cuda()
        visualX = visualX.flatten(start_dim=0, end_dim=1).cuda()
        labels = labels.flatten(start_dim=0, end_dim=1).cuda()

        # forward
        outsAV, outsA, outsV = model(audioX, visualX)

        lossAV = criterion(outsAV, labels)
        lossA = criterion(outsA, labels)
        lossV = criterion(outsV, labels)
        loss = lossAV + 0.4 * lossA + 0.4 * lossV
        acc = accuracy(outsAV, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record
        train_stats.update('Loss', loss.item(), audioX.size(0))
        train_stats.update('Acc', acc.item(), audioX.size(0))
        train_stats.update('Lr', optimizer.param_groups[0]["lr"])
        train_stats.update('Time', time.time() - end)

        if config.rank == 0 and i % config.log_batch_freq == 0:
            logger.info(progress.display(i))

        end = time.time()

    key_stats={
        'Avg_loss': train_stats.avg('Loss'),
        'Avg_acc': train_stats.avg('Acc'),
        'Lr_value': train_stats.val('Lr')
    }

    return key_stats

def evaluate(val_loader, model, criterion, epoch, logger, config):
    val_stats = AverageMeters()
    val_stats.add('Time', ':6.3f')
    val_stats.add('Data', ':6.3f')
    val_stats.add('Loss', ':.4e')
    val_stats.add('Acc', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        val_stats,
        prefix="Epoch: [{}]".format(epoch)
    )

    # eval mode
    model.eval()

    end = time.time()
    pred_scores_list = [None for _ in range(config.world_size)]
    ground_labels_list = [None for _ in range(config.world_size)]
    pred_scores, ground_labels = torch.tensor([]), torch.tensor([])
    if hasattr(model, 'module'):
        model = model.module
    with torch.no_grad():
        for i, (audioX, visualX, labels) in enumerate(val_loader):
            audioX = audioX.flatten(start_dim=0, end_dim=1).cuda()
            visualX = visualX.flatten(start_dim=0, end_dim=1).cuda()
            labels = labels.flatten(start_dim=0, end_dim=1).cuda()

            # forward
            outsAV, outsA, outsV = model(audioX, visualX)

            # compute scores
            pred_score = torch.softmax(outsAV, dim=-1)[..., 1]
            pred_scores = torch.cat((pred_scores, pred_score.detach().reshape(-1).cpu()))
            ground_labels = torch.cat((ground_labels, labels.reshape(-1).cpu()))

            lossAV = criterion(outsAV, labels)
            lossA = criterion(outsA, labels)
            lossV = criterion(outsV, labels)
            loss = lossAV + 0.4 * lossA + 0.4 * lossV

            acc = accuracy(outsAV, labels)

            # record
            val_stats.update('Loss', loss.item(), audioX.size(0))
            val_stats.update('Acc', acc.item(), audioX.size(0))
            val_stats.update('Time', time.time() - end)

            if config.rank == 0 and i % config.log_batch_freq == 0:
                logger.info(progress.display(i))

            end = time.time()

    # gather results
    dist.all_gather_object(pred_scores_list, pred_scores.numpy())
    dist.all_gather_object(ground_labels_list, ground_labels.numpy())

    # compute mAP
    mAP = average_precision(
        np.concatenate(pred_scores_list),
        np.concatenate(ground_labels_list)
        )
    key_stats={
        'Avg_loss': val_stats.avg('Loss'),
        'Avg_acc': val_stats.avg('Acc'),
        'mAP': mAP*100,
    }

    return key_stats

def test_process(config):
    dist.init_process_group(backend='gloo')
    logger = get_logger()

    # dataset
    test_dataset = build('val_dataset', config)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    config.val_dataloader['args']['sampler'] = test_sampler
    test_dataloader = build('val_dataloader', config)

    # model
    model = build('model', config)
    model.cuda()

    # loss function
    criterion = build('loss', config)

    # others
    config.checkpointer['args']['recoverables'] = {'model':model}
    checkpointer = build('checkpointer', config)
    checkpointer.recover_if_possible(epoch=config.num_epoch, device='cuda')

    # test
    key_stats = evaluate(test_dataloader, model, criterion, config.num_epoch, logger, config)
    if config.rank == 0:
        logger.info('mAP: %6.2f%%'%(key_stats['mAP']))

if __name__ == '__main__':
    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, overrides, True)
    config.rank = int(os.environ['LOCAL_RANK'])
    config.world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(args.gpu[config.rank % len(args.gpu)])
    torch.cuda.set_device(gpu)

    if args.test:
        # testing
        test_process(config)
    else:
        # training
        train_process(config)
