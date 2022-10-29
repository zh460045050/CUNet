import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from models.basic import *


class SharedDecoder(nn.Module):
    def __init__(self, args):
        super(SharedDecoder,self).__init__()

        self.args = args

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up2 = up_conv(ch_in=128,ch_out=64)

        if 'Att' in self.args.model_type:
            self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
            self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
            self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
            self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)

        if 'R2' in self.args.model_type:
            self.Up_conv5 = RRCNN_block(ch_in=1024, ch_out=512, t=self.args.t)
            self.Up_conv4 = RRCNN_block(ch_in=512, ch_out=256, t=self.args.t)
            self.Up_conv3 = RRCNN_block(ch_in=256, ch_out=128, t=self.args.t)
            self.Up_conv2 = RRCNN_block(ch_in=128, ch_out=64, t=self.args.t)
        else:
            self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
            self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
            self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
            self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.convg = single_conv(ch_in=64, ch_out=64)

    def forward(self,xs):
        # encoding path

        [x5, x4, x3, x2, x1] = xs

        # decoding + concat path
        d5 = F.interpolate(x5,(x4.size(2),x4.size(3)),mode='bilinear')
        d5 = self.Up5(d5)
        if 'Att' in self.args.model_type:
            x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = F.interpolate(d5,(x3.size(2),x3.size(3)),mode='bilinear')
        d4 = self.Up4(d4)
        if 'Att' in self.args.model_type:
            x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = F.interpolate(d4,(x2.size(2),x2.size(3)),mode='bilinear')
        d3 = self.Up3(d3)
        if 'Att' in self.args.model_type:
            x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = F.interpolate(d3,(x1.size(2),x1.size(3)),mode='bilinear')
        d2 = self.Up2(d2)
        if 'Att' in self.args.model_type:
            x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        fs = self.convg(d2)

        return fs, [d2, d3, d4, d5]

