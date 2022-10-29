import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from models.basic import *
from models.decoders.specific_decoder import SpecificDecoder
from models.encoders.ushape_encoder import SharedEncoder
from models.decoders.ushape_decoder import SharedDecoder


class MU_Net(nn.Module):
    def __init__(self, args):
        super(MU_Net,self).__init__()
        
        self.args = args
        self.shared_encoder = SharedEncoder(args)
        self.shared_decoder = SharedDecoder(args)

        if self.args.ms_mode != 'only_global':
            print("..........Warining! Only Using the Global Structure..........")
            self.blood_decoder = SpecificDecoder(args)
            self.choroid_decoder = SpecificDecoder(args)

        if self.args.ms_mode == '':
            self.conv_blood = nn.Conv2d(128, self.args.output_ch ,kernel_size=1,stride=1,padding=0)
            self.conv_choroid = nn.Conv2d(128, self.args.output_ch ,kernel_size=1,stride=1,padding=0)
        else:
            print("..........Warining! Only Using the Global or Specifical Structure..........")
            self.conv_blood = nn.Conv2d(64, self.args.output_ch ,kernel_size=1,stride=1,padding=0)
            self.conv_choroid = nn.Conv2d(64, self.args.output_ch ,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        
        xs = self.shared_encoder(x)
        fs, dss = self.shared_decoder(xs)

        if self.args.ms_mode != 'only_global':
            fd_c, dcs = self.choroid_decoder(xs, dss)
            fd_b, dbs = self.blood_decoder(xs, dss)
        else:
            fd_c = fs
            fd_b = fs
        
        if self.args.ms_mode == '':
            blood = self.conv_blood(torch.cat((fd_b,fs), dim=1))
            choroid = self.conv_choroid(torch.cat((fd_c,fs), dim=1))
        elif self.args.ms_mode == 'only_global':
            blood = self.conv_blood(fs)
            choroid = self.conv_choroid(fs)
        elif self.args.ms_mode == 'only_specific':
            blood = self.conv_blood(fd_b)
            choroid = self.conv_choroid(fd_c)

        return [blood, choroid], fs, fd_c, fd_b

