# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.distributed as dist
import speakerlab.utils.utils_rdino as utils_rdino

class SDPNLoss(nn.Module):
    def __init__(self, tau=0.1, me_max=True):
        super().__init__()
        self.tau = tau
        self.me_max = me_max
        self.softmax = torch.nn.Softmax(dim=1)

    def sharpen(self, p, T):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def snn(self, query, supports, support_labels):
        """ Soft Nearest Neighbours similarity classifier """
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)
        return self.softmax(query @ supports.T / self.tau) @ support_labels


    def forward(self, anchor_views, target_views, prototypes, proto_labels, T=0.25, use_sinkhorn=True):

        # Step 1: compute anchor predictions
        probs = self.snn(anchor_views, prototypes, proto_labels)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = self.sharpen(self.snn(target_views, prototypes, proto_labels), T=T)
            if use_sinkhorn:
                targets = self.distributed_sinkhorn(targets)
            # targets = torch.cat([targets for _ in range(num_views)], dim=0)
            targets = torch.cat([targets for _ in range(4)], dim=0)

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if self.me_max:
            avg_probs = utils_rdino.AllReduce.apply(torch.mean(probs, dim=0))
            rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))

        # -- logging
        with torch.no_grad():
            num_ps = float(len(set(targets.argmax(dim=1).tolist())))
            max_t = targets.max(dim=1).values.mean()
            min_t = targets.min(dim=1).values.mean()
            log_dct = {'np': num_ps, 'max_t': max_t, 'min_t': min_t}

        return loss, rloss, log_dct, targets


    @torch.no_grad()
    def distributed_sinkhorn(self, Q, num_itr=3, use_dist=True):
        _got_dist = use_dist and torch.distributed.is_available() \
            and torch.distributed.is_initialized() \
            and (torch.distributed.get_world_size() > 1)

        if _got_dist:
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1

        Q = Q.T
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if _got_dist:
            torch.distributed.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(num_itr):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if _got_dist:
                torch.distributed.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.T


