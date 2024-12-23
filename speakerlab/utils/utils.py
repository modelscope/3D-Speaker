# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import sys
import os
import random
import logging
import yaml
import numpy as np
from contextlib import contextmanager

import torch
from speakerlab.utils.fileio import load_yaml

def parse_config(config_file):
    if config_file.endwith('.yaml'):
        config = load_yaml(config_file)
    else:
        raise Exception("Other formats not currently supported.")
    return config

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

def get_logger(fpath=None, fmt=None):
    if fmt is None:
        fmt = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if fpath is not None:
        handler = logging.FileHandler(fpath)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger

def get_utt2spk_dict(utt2spk, suffix=''):
    temp_dict={}
    with open(utt2spk,'r') as utt2spk_f:
        lines = utt2spk_f.readlines()
    for i in lines:
        i=i.strip().split()
        if suffix == '' or suffix is None:
            key_i = i[0]
            value_spk = i[1]
        else:
            key_i = i[0]+'_'+suffix
            value_spk = i[1]+'_'+suffix
        if key_i in temp_dict:
            raise ValueError('The key must be unique.')
        temp_dict[key_i]=value_spk
    return temp_dict

def get_wavscp_dict(wavscp, suffix=''):
    temp_dict={}
    with open(wavscp, 'r') as wavscp_f:
        lines = wavscp_f.readlines()
    for i in lines:
        i=i.strip().split()
        if suffix == '' or suffix is None:
            key_i = i[0]
        else:
            key_i = i[0]+'_'+suffix
        value_path = i[1]
        if key_i in temp_dict:
            raise ValueError('The key must be unique.')
        temp_dict[key_i]=value_path
    return temp_dict

def accuracy(x, target):
    # x: [*, C], target: [*,]
    _, pred = x.topk(1)
    pred = pred.squeeze(-1)
    acc = pred.eq(target).float().mean()
    return acc*100

def average_precision(scores, labels):
    # scores: [N, ], labels: [N, ]
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if isinstance(scores, list):
        scores = np.array(scores)
    if isinstance(labels, list):
        labels = np.array(labels)
    assert isinstance(scores, np.ndarray) and isinstance(
        labels, np.ndarray), 'Input should be numpy.array.'
    assert len(scores.shape)==1 and len(labels.shape)==1 and \
        scores.shape[0]==labels.shape[0]

    sort_idx = np.argsort(scores)[::-1]
    scores = scores[sort_idx]
    labels = labels[sort_idx]
    tp_count = (labels==1).sum()
    tp = labels.cumsum()
    recall = tp / tp_count
    precision = tp / (np.arange(len(labels)) + 1)

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Smooth precision to be monotonically decreasing.
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision

def load_params(dst_model, src_state, strict=True):
    dst_state = {}
    for k in src_state:
        if k.startswith('module'):
            dst_state[k[7:]] = src_state[k]
        else:
            dst_state[k] = src_state[k]
    dst_model.load_state_dict(dst_state, strict=strict)
    return dst_model

def merge_vad(vad1: list, vad2: list):
    intervals = vad1 + vad2
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class AverageMeters(object):
    def __init__(self, names: list = None, fmts: list = None):
        self.cont = dict()
        if names is None or fmts is None:
            return
        for name, fmt in zip(names, fmts):
            self.cont[name] = AverageMeter(name, fmt)

    def add(self, name, fmt=':f'):
        self.cont[name] = AverageMeter(name, fmt)

    def update(self, name, val, n=1):
        self.cont[name].update(val, n)

    def avg(self, name):
        return self.cont[name].avg

    def val(self, name):
        return self.cont[name].val

    def __str__(self):
        return '\t'.join([str(s) for s in self.cont.values()])


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(self.meters)]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@contextmanager
def silent_print():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

def download_model_from_modelscope(model_id, model_revision=None, cache_dir=None):
    from modelscope.hub.snapshot_download import snapshot_download
    if cache_dir is None:
        cache_dir = snapshot_download(
            model_id,
            revision=model_revision,
        )
    else:
        cfg_file = os.path.join(cache_dir, model_id, 'configuration.json')
        if not os.path.exists(cfg_file):
            cache_dir = snapshot_download(
                model_id,
                revision=model_revision,
                cache_dir=cache_dir,
            )
        else:
            cache_dir = os.path.join(cache_dir, model_id)
    return cache_dir

def circle_pad(x: torch.Tensor, target_len, dim=0):
    xlen = x.shape[dim]
    if xlen >= target_len:
        return x
    n = int(np.ceil(target_len/xlen))
    xcat = torch.cat([x for _ in range(n)], dim=dim)
    return torch.narrow(xcat, dim, 0, target_len)
