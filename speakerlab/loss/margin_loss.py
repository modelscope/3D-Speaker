# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcMarginLoss(nn.Module):
    """
    Implement of additive angular margin loss.
    """
    def __init__(self,
                 scale=32.0,
                 margin=0.2,
                 easy_margin=False):
        super(ArcMarginLoss, self).__init__()
        self.scale = scale
        self.easy_margin = easy_margin
        self.criterion = nn.CrossEntropyLoss()

        self.update(margin)

    def forward(self, cosine, label):
        # cosine : [batch, numclasses].
        # label : [batch, ].
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = torch.zeros(cosine.size()).type_as(cosine)
        one_hot.scatter_(1, label.unsqueeze(1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.criterion(output, label)
        return loss

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)


class AddMarginLoss(nn.Module):
    """
    Implement of additional  margin loss.
    """
    def __init__(self,
                 scale=32.0,
                 margin=0.2,
                 easy_margin=False):
        super(AddMarginLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.criterion = nn.CrossEntropyLoss()

        self.update(margin)

    def forward(self, cosine, label):
        # cosine : [batch, numclasses].
        # label : [batch, ].
        phi = cosine - self.margin
        one_hot = torch.zeros(cosine.size()).type_as(cosine)
        one_hot.scatter_(1, label.unsqueeze(1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.criterion(output, label)
        return loss

    def update(self, margin=0.2):
        self.margin = margin


class EntropyLoss(nn.Module):
    def __init__(self,**kwargs):
        super(EntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        # x : [*, numclasses].
        # label : [*, ].
        assert x.dim() == label.dim()+1
        x = x.flatten(start_dim=0, end_dim=x.dim()-2)
        label = label.flatten(start_dim=0, end_dim=label.dim()-1)
        loss = self.criterion(x, label)
        return loss

    def update(self, margin=None):
        pass
