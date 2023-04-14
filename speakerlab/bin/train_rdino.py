# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import json
import math
import numpy
import argparse
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import speakerlab.utils.utils as utils
import speakerlab.utils.utils_rdino as utils_rdino
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build

parser = argparse.ArgumentParser(description='Regularized DINO Framework Training')
parser.add_argument('--config', default='', type=str, help='Config file')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')

def main():
    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, overrides, True)

    # DDP
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(args.gpu[rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl")

    utils_rdino.setup_for_distributed(rank == 0)
    utils.set_seed(args.seed)

    model_save_path = config.exp_dir + "/models"
    log_save_path = config.exp_dir

    if rank == 0:
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(log_save_path, exist_ok=True)

    # Initialise data loader
    train_dataset = build('dataset', config)
    sampler = torch.utils.data.DistributedSampler(train_dataset)
    # build dataloader
    config.dataloader['args']['sampler'] = sampler
    train_loader = build('dataloader', config)

    print(f"Data loaded: there are {len(train_loader)} iterations.")

    # Load models
    student = build('student_model', config)
    teacher = build('teacher_model', config)

    # synchronize batch norms
    student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
    student.cuda()
    teacher.cuda()

    # DDP wrapper
    teacher = nn.parallel.DistributedDataParallel(teacher)
    student = nn.parallel.DistributedDataParallel(student)

    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print("Student and teacher are using the same network architecture but different parameters.")

    # Prepare dino loss
    dino_loss = build('dino_loss', config)
    dino_loss.cuda()

    # Prepare regularized terms loss
    regularized_loss = build('regularized_loss', config)
    regularized_loss.cuda()

    # Prepare optimizer
    params_groups = utils_rdino.get_params_groups(student)
    optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)

    # Prepare learning rate schedule and weight decay scheduler
    lr_schedule = utils_rdino.cosine_scheduler(
        config.lr * (config.batch_size_per_gpu * utils_rdino.get_world_size()) / 256.,
        config.min_lr,
        config.epochs,
        len(train_loader),
        warmup_epochs=config.warmup_epochs,
    )

    wd_schedule = utils_rdino.cosine_scheduler(
        config.weight_decay,
        config.weight_decay_end,
        config.epochs,
        len(train_loader),
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils_rdino.cosine_scheduler(
        config.momentum_teacher,
        1,
        config.epochs,
        len(train_loader)
    )

    # Resume training optionally
    to_restore = {"epoch": 0}
    utils_rdino.restart_from_checkpoint(
        os.path.join(model_save_path, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting RDINO training !")

    for epoch in range(start_epoch, config.epochs):
        train_loader.sampler.set_epoch(epoch)
        # Training one epoch
        train_stats = train_one_epoch(student, teacher, dino_loss, regularized_loss, train_loader, optimizer, \
            lr_schedule, wd_schedule, momentum_schedule, epoch, config)

        # Write logs
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'dino_loss': dino_loss.state_dict(),
        }

        utils_rdino.save_on_master(save_dict, os.path.join(model_save_path, 'checkpoint.pth'))
        if config.saveckp_freq and epoch % config.saveckp_freq == 0:
            utils_rdino.save_on_master(save_dict, os.path.join(model_save_path, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils_rdino.is_main_process():
            with  open(log_save_path + "/log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")


def train_one_epoch(student, teacher, dino_loss, regularized_loss, train_loader, optimizer, lr_schedule, \
                    wd_schedule, momentum_schedule, epoch, config):

    teacher.train()
    student.train()

    metric_logger = utils_rdino.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch+1, config.epochs)
    for it, data in enumerate(
        metric_logger.log_every(
            train_loader, print_freq=100, header=header)):
        with torch.autograd.set_detect_anomaly(True):
            # update weight decay and learning rate according to their schedule
            it = len(train_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            # Data.shape is [batch_size, glb_num + int(local_num / 2), fbank_dim, frames]
            # Local_frames is half of the global_frames (max_frames)
            batch_size, _, fbank_dim, frame_dim = data.shape
            data = data.transpose(0,1).cuda()
            global_data = data[0: config.glb_num, :, :, :]
            global_data = global_data.reshape(-1, fbank_dim, frame_dim)
            tea_reg_out, teacher_output = teacher(global_data)
            stu_reg_out, student_global_output = student(global_data)

            local_data1 = data[config.glb_num, :, :, :]
            local_data2 = data[config.glb_num + 1, :, :, :]
            local_frames = math.floor(config.max_frames / 2)
            local_data1_1 = local_data1[:, :, 0: local_frames]
            local_data1_2 = local_data1[:, :, local_frames: ]
            local_data2_1 = local_data2[:, :, 0: local_frames]
            local_data2_2 = local_data2[:, :, local_frames:]
            local_data = torch.cat((local_data1_1,local_data1_2,local_data2_1,local_data2_2), axis=0)

            _, student_local_output = student(local_data)
            student_output = torch.cat((student_global_output, student_local_output), axis=0)

            d_loss = dino_loss(student_output, teacher_output, epoch)
            reg_loss = regularized_loss(tea_reg_out, stu_reg_out)

            begin_lambd = numpy.linspace(0.05, config.lambd, 10)
            later_lambd = numpy.linspace(config.lambd, config.lambd, 60)
            lambd_sche = numpy.concatenate((begin_lambd, later_lambd), axis=0)
            lambd_cur = lambd_sche[epoch]

            loss = d_loss + lambd_cur * reg_loss

            if not math.isfinite(loss.item()):
                raise Exception("Loss is {}, stopping training".format(loss.item()), force=True)

            # Student network update
            optimizer.zero_grad()
            param_norms = None
            loss.backward()
            if config.clip_grad:
                param_norms = utils_rdino.clip_gradients(student, config.clip_grad)
            utils_rdino.cancel_gradients_last_layer(epoch, student, config.freeze_last_layer)
            optimizer.step()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss_dino=d_loss.item())
            metric_logger.update(loss_reg=reg_loss.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()
