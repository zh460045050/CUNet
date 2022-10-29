import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from libs.evaluation import *
from models.msunet import MU_Net
import csv
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import mark_boundaries
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from tensorboardX import SummaryWriter

def SegLoss(probs, GT):

    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.BCELoss()
    b, c, h, w = probs.size()
    output_ch = c
    c = 1

    prob_flat = probs[:, 1, :, :].unsqueeze(1).permute(0, 2, 3, 1).contiguous().view(-1, c)
    W = torch.ones(GT.size()).cuda()
    W[GT == 0] = 1 / torch.sum(GT == 0, dim=[0,1,2])
    W[GT == 1] = 1 / torch.sum(GT == 1, dim=[0,1,2])
    W = W.view(-1, c)
    GT_flat = GT.view(GT.size(0),-1).cpu().data
    GT_flat = torch.from_numpy(to_categorical(GT_flat, output_ch)).to(devices).view(-1, output_ch)

    loss = criterion(prob_flat, GT_flat[:, 1].unsqueeze(-1).float()) * W 
    loss = torch.sum(loss)

    return loss