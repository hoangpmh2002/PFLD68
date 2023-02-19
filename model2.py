from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional
import math
from collections import OrderedDict
from resnest.torch import resnest50
class MyResNest50(nn.Module):
    
    def __init__(self, nums_class=136):
        super(MyResNest50, self).__init__()

        self.resnest = resnest50(pretrained=True)
        self.resnest_backbone1 = nn.Sequential(*list(self.resnest.children())[:-6])
        self.resnest_backbone_end = nn.Sequential(*list(self.resnest.children())[-6:-2])
        
        self.in_features = 2048 * 4 * 4
        self.fc = nn.Linear(in_features=self.in_features, out_features=nums_class)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        auxnet = self.resnest_backbone1(x)
        #print(auxnet.size())
        out = self.resnest_backbone_end(auxnet)
        #print(out.size())
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, auxnet

