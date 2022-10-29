import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from models.basic import *


class SharedEncoder(nn.Module):
    def __init__(self, args):
        super(SharedEncoder,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.args = args

        if 'R2' in self.args.model_type:
            self.Conv1 = RRCNN_block(ch_in=1,ch_out=64,t=self.args.t)
            self.Conv2 = RRCNN_block(ch_in=64,ch_out=128,t=self.args.t)
            self.Conv3 = RRCNN_block(ch_in=128,ch_out=256,t=self.args.t)
            self.Conv4 = RRCNN_block(ch_in=256,ch_out=512,t=self.args.t)
            self.Conv5 = RRCNN_block(ch_in=512,ch_out=1024,t=self.args.t)
        else:
            self.Conv1 = conv_block(ch_in=1,ch_out=64)
            self.Conv2 = conv_block(ch_in=64,ch_out=128)
            self.Conv3 = conv_block(ch_in=128,ch_out=256)
            self.Conv4 = conv_block(ch_in=256,ch_out=512)
            self.Conv5 = conv_block(ch_in=512,ch_out=1024)

    def forward(self,x):

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        return [x5, x4, x3, x2, x1]

