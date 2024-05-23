# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

""" This TDNN implementation is adapted from https://github.com/wenet-e2e/wespeaker.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import speakerlab.models.eres2net.pooling_layers as pooling_layers


class Tdnn_layer(nn.Module):

    def __init__(self, in_dim, out_dim, context_size, dilation=1, padding=0):
        """Define the TDNN layer, essentially 1-D convolution

        Args:
            in_dim (int): input dimension
            out_dim (int): output channels
            context_size (int): context size, essentially the filter size
            dilation (int, optional):  Defaults to 1.
            padding (int, optional):  Defaults to 0.
        """
        super(Tdnn_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_size = context_size
        self.dilation = dilation
        self.padding = padding
        self.conv_1d = nn.Conv1d(self.in_dim,
                                 self.out_dim,
                                 self.context_size,
                                 dilation=self.dilation,
                                 padding=self.padding)

        # Set Affine=false to be compatible with the original kaldi version
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        out = self.conv_1d(x)
        out = F.relu(out)
        out = self.bn(out)
        return out


class Xvector(nn.Module):

    def __init__(self,
                 feat_dim=40,
                 hid_dim=512,
                 stats_dim=1500,
                 embed_dim=512,
                 pooling_func='TSTP'):
        
        super(Xvector, self).__init__()
        self.feat_dim = feat_dim
        self.stats_dim = stats_dim
        self.embed_dim = embed_dim

        self.frame_1 = Tdnn_layer(feat_dim, hid_dim, context_size=5, dilation=1)
        self.frame_2 = Tdnn_layer(hid_dim, hid_dim, context_size=3, dilation=2)
        self.frame_3 = Tdnn_layer(hid_dim, hid_dim, context_size=3, dilation=3)
        self.frame_4 = Tdnn_layer(hid_dim, hid_dim, context_size=1, dilation=1)
        self.frame_5 = Tdnn_layer(hid_dim,
                                 stats_dim,
                                 context_size=1,
                                 dilation=1)

        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == "TSDP" else 2
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=self.stats_dim)
        self.seg_1 = nn.Linear(self.stats_dim * self.n_stats, embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) -> (B,F,T)

        out = self.frame_1(x)
        out = self.frame_2(out)
        out = self.frame_3(out)
        out = self.frame_4(out)
        out = self.frame_5(out)

        stats = self.pool(out)
        embed_a = self.seg_1(stats)

        return embed_a


if __name__ == '__main__':

    x = torch.zeros(10, 300, 80)
    model = Xvector(feat_dim=80, embed_dim=512)
    model.eval()
    out = model(x)
    print(out.shape) # torch.Size([10, 512])

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6)) # 4.34 M
