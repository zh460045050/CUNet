from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io as io
import cv2

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

import sys
import errno
import shutil
import json
import os.path as osp
from skimage import color
from skimage.segmentation._slic import _enforce_label_connectivity_cython

import torchvision

import random
random.seed(1)
torch.manual_seed(1)

def mk_save_dir_val(root_path, epoch):
    cur_save_path = root_path + 'validation_epoch_' + str(epoch)
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
    
    f_path = cur_save_path + "/vis_feature"
    if not os.path.exists(f_path):
        os.makedirs(f_path)

    rs_path = cur_save_path + "/vis_result"
    if not os.path.exists(rs_path):
        os.makedirs(rs_path)

    gd_path = root_path + "/vis_gd"
    if not os.path.exists(gd_path):
        os.makedirs(gd_path)

    gd_blood_path = gd_path + "/blood"
    gd_choroid_path = gd_path + "/choroid"
    if not os.path.exists(gd_blood_path):
        os.makedirs(gd_blood_path)
    if not os.path.exists(gd_choroid_path):
        os.makedirs(gd_choroid_path)

    rs_blood_path = rs_path + "/blood"
    rs_choroid_path = rs_path + "/choroid"
    if not os.path.exists(rs_blood_path):
        os.makedirs(rs_blood_path)
    if not os.path.exists(rs_choroid_path):
        os.makedirs(rs_choroid_path)

    return cur_save_path, f_path, rs_path, gd_path, gd_blood_path, gd_choroid_path, rs_blood_path, rs_choroid_path


def save_features(feature, save_dir, filename, mode='mean', type='feature'):

    if mode == 'mean':
        feature = F.sigmoid(feature)
        feature = torch.mean(feature, dim=0)

    torchvision.utils.save_image(feature.data.cpu(),
        os.path.join(save_dir,
            '%s_%s.png'%(filename, type)))

def tensor2image(img):

    trans_img = img.expand(3, img.size(1), img.size(2)) * 0.5  + 0.5
    trans_img = trans_img.permute(1, 2, 0).data.cpu()

    return trans_img

def save_segmentation(labels, img, save_dir, filename, type='blood'):

    gd_img = mark_boundaries(img, np.int32(labels))
    plt.imsave(os.path.join(save_dir,
                                    '%s_%s.png'%(filename, type)), gd_img)

class Logger(object):
    """
        Write console output to external text file.
        Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
        """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()
        
    def __enter__(self):
        pass
        
    def __exit__(self, *args):
        self.close()
        
    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
        
    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

