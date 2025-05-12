# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


### Built from https://github.com/kuldeepbrd1/image-corruptions





import numpy as np
from PIL import Image
import pickle


# /////////////// Corruption Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from skimage import transform, feature


import warnings
import os

import torch
import copy

warnings.simplefilter("ignore", UserWarning)

CORRUPTIONS = ['identity',
                'impulse_noise',
                'glass_blur',
                'motion_blur',
                'rotate',
                'spatter',]

ALL_CORRUPTIONS = ['identity',
               'gaussian_noise',
               'impulse_noise',
               'glass_blur',
               'spatter',
               'saturate',
               'rotate',]



# /////////////// Corruptions ///////////////

def identity(x):
    return np.array(x, dtype=np.float32)




def gaussian_noise(x, severity=5):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    # x = copy.deepcopy(x) + torch.normal(0, c, size=x.size())
    norms = torch.normal(0, c, size=x.size())
    x = copy.deepcopy(x).detach().cpu() + norms
    return x.cuda()



def impulse_noise(x, severity=4):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    
    x = copy.deepcopy(x).detach().cpu().numpy()
    x = sk.util.random_noise(x, mode='s&p', amount=c, clip=False)
    
    x = np.clip(x, 0, 1)

    return torch.from_numpy(x)



def gaussian_blur(x, severity=2):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = copy.deepcopy(x).detach().cpu().numpy()

    x = gaussian(x, sigma=c, channel_axis=0)
    x = np.clip(x, 0, 1)
    return torch.from_numpy(x)



def spatter(x, severity=4):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]

    x = copy.deepcopy(x).detach().cpu().numpy()

    liquid_layer = np.random.normal(size=x.shape, loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    
    m = np.where(liquid_layer > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0

    # mud spatter
    color = 63 / 255. * np.ones_like(x) * m
    x *= (1 - m)
    x = np.clip(x + color, 0, 1)

    return torch.from_numpy(x)




def saturate(x, severity=5):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = copy.deepcopy(x).detach().cpu().numpy()
    x = sk.color.rgb2hsv(x,channel_axis=0)
    x = np.clip(x * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x, channel_axis=0)

    x = np.clip(x, 0, 1)
    return torch.from_numpy(x)



def rotate(x, severity=2):
    c = [0.2, 0.4, 0.6, 0.8, 1.][severity-1]

    # Randomly switch directions
    bit = np.random.choice([-1, 1], 1)[0]
    c *= bit
    aff = transform.AffineTransform(rotation=c)

    a1, a2 = aff.params[0,:2]
    b1, b2 = aff.params[1,:2]

    ### 32 used as the midpoint for 64x64 synthetic images
    a3 = 32 * (1 - a1 - a2)
    b3 = 32 * (1 - b1 - b2)
    # a3 = 13.5 * (1 - a1 - a2)
    # b3 = 13.5 * (1 - b1 - b2)
    aff = transform.AffineTransform(rotation=c, translation=[a3, b3])

    x = copy.deepcopy(x).detach().cpu().numpy()
    x = transform.warp(np.transpose(x, (1, 2, 0)), inverse_map=aff)
    x = np.clip(x, 0, 1)
    return torch.from_numpy(np.transpose(x, (2, 0, 1)))
    
    