import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from models.basic import *


class SpecificDecoder(nn.Module):
    def __init__(self, args):
        super(SpecificDecoder,self).__init__()

        self.args = args

        self.Up5 = single_conv(ch_in=1024, ch_out=512)
        self.Up4 = single_conv(ch_in=512, ch_out=256)
        self.Up3 = single_conv(ch_in=256, ch_out=128)
        self.Up2 = single_conv(ch_in=128, ch_out=64)

        self.Up_conv5 = single_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = single_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = single_conv(ch_in=128, ch_out=64)

    def forward(self, xs, ds):
        # encoding path

        [x5, x4, x3, x2, x1] = xs
        [d2, d3, d4, d5] = ds

        d_s5 = torch.cat((x4,d5),dim=1)
        d_s5 = self.Up5(d_s5)

        d_s4 = F.interpolate(d_s5,(d4.size(2), d4.size(3)),mode='bilinear')
        d_s4 = self.Up_conv5(d_s4)
        d_s4 = torch.cat((d_s4,d4),dim=1)
        d_s4 = self.Up4(d_s4)

        d_s3 = F.interpolate(d_s4,(d3.size(2), d3.size(3)),mode='bilinear')
        d_s3 = self.Up_conv4(d_s3)
        d_s3 = torch.cat((d_s3, d3),dim=1)
        d_s3 = self.Up3(d_s3)

        d_s2 = F.interpolate(d_s3, (d2.size(2), d2.size(3)),mode='bilinear')
        d_s2 = self.Up_conv3(d_s2)
        d_s2 = torch.cat((d_s2,d2),dim=1)
        fd = self.Up2(d_s2)

        return fd, [d_s2, d_s3, d_s4, d_s5]

