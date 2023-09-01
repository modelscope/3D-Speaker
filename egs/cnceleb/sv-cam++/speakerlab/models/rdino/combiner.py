# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch

class Combiner(torch.nn.Module):
    """
    Combine backbone (ECAPA) and head (MLP)
    """
    def __init__(self, backbone, head):
        super(Combiner, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        output = self.head(x)
        return output
