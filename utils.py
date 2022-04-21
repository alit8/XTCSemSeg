import os
import time
import numpy as np
import random

import torch
from torch import nn

import numpy as np
# import pydensecrf.densecrf as dcrf
# import pydensecrf.utils as utils

# class DenseCRF(object):
#     def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
#         self.iter_max = iter_max
#         self.pos_w = pos_w
#         self.pos_xy_std = pos_xy_std
#         self.bi_w = bi_w
#         self.bi_xy_std = bi_xy_std
#         self.bi_rgb_std = bi_rgb_std

#     def __call__(self, image, probmap):
#         C, H, W = probmap.shape

#         U = utils.unary_from_softmax(probmap)
#         U = np.ascontiguousarray(U)

#         image = np.ascontiguousarray(image)

#         d = dcrf.DenseCRF2D(W, H, C)
#         d.setUnaryEnergy(U)
#         d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
#         d.addPairwiseBilateral(
#             sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
#         )

#         Q = d.inference(self.iter_max)
#         Q = np.array(Q).reshape((C, H, W))

#         return Q

BASE_DIR = "."

MODELS_DIR = f"{BASE_DIR}/models"
DATA_DIR = "../ADE20K"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints"
LOG_DIR = f"{BASE_DIR}/logs"

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def get_device(cuda):
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")

    return device

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # cpu  vars
    torch.cuda.manual_seed_all(seed) # gpu vars

def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)