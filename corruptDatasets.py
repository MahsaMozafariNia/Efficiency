"""
Does standard subnetwork training on all tasks

"""

from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import warnings
import copy
import time
import torch.nn as nn
import numpy as np
import torch
from itertools import islice
from torch.optim.lr_scheduler  import MultiStepLR
from math import floor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torch.utils.data as D
import random

from AuxiliaryScripts import utils, cldatasets, corruptions
from AuxiliaryScripts.manager import Manager
from AuxiliaryScripts.Normalization_metrics import Normalization_Techniques
from AuxiliaryScripts.RemovalMetrics import HSIC, energy_removal, entropy_removal, cosine_method 
from AuxiliaryScripts.RemovalMetrics.Caper.Caper import Caper_Method
from AuxiliaryScripts.RemovalMetrics import EpochAcc
from AuxiliaryScripts.RemovalMetrics import TauAcc


# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--run_id', type=str, default="000", help='Id of current run.')
FLAGS.add_argument('--arch', choices=['resnet18', 'modresnet18', 'resnet50', 'vgg16', 'vgg16_new'], default='resnet18', help='Architectures')
FLAGS.add_argument('--pretrained', action='store_true', default=False, help='Whether or not to load a predefined pretrained state dict in Network().')
FLAGS.add_argument('--load_from', choices=['baseline', 'steps'], default='baseline', help='Whether or not we are loading from the baseline')
FLAGS.add_argument('--task_num', type=int, default=0, help='Current task number.')

FLAGS.add_argument('--dataset', type=str, choices=['MPC', 'KEF', 'MPCoffset', 'splitcifar', 'TIC', 'SynthJoint', 'SynthDisjoint'], default='splitcifar', help='Name of dataset')
FLAGS.add_argument('--dataset_modifier', choices=['None', 'CIFAR100Full', 'OnlyCIFAR100', 'ai', 'nature'], default='None', help='Overloaded parameter for various adjustments to dataloaders in utils')
FLAGS.add_argument('--preprocess', choices=['Normalized', 'Unnormalized'], default='Unnormalized', help='Determines if the data is ranged 0:1 unnormalized or not (normalized')

FLAGS.add_argument('--attack_type', choices=['PGD', 'AutoAttack', 'gaussian_noise', 'impulse_noise', 'gaussian_blur', 'spatter', 'saturate', 'rotate'], default='PGD', help='What type of perturbation is applied')
FLAGS.add_argument('--removal_metric',  type=str , default='Random', choices=['Caper', 'Random', 'NoRemoval', 'EpochAcc'], help='which metric to use for removing training samples')
FLAGS.add_argument('--trial_num', type=int , default=1, help='Trial number for setting manual seed')



# Training options.
FLAGS.add_argument('--use_train_scheduler', action='store_true', default=False, help='If true will train with a fixed lr schedule rather than early stopping and lr decay based on validation accuracy.')
FLAGS.add_argument('--train_epochs', type=int, default=2, help='Number of epochs to train for')
FLAGS.add_argument('--eval_interval', type=int, default=5, help='The number of training epochs between evaluating accuracy')
FLAGS.add_argument('--batch_size', type=int, default=128, help='Batch size')
FLAGS.add_argument('--lr', type=float, default=0.1, help='Learning rate')
FLAGS.add_argument('--lr_min', type=float, default=0.001, help='Minimum learning rate below which training is stopped early')
FLAGS.add_argument('--lr_patience', type=int, default=5, help='Patience term to dictate when Learning rate is decreased during training')
FLAGS.add_argument('--lr_factor', type=float, default=0.1, help='Factor by which to reduce learning rate during training')
FLAGS.add_argument('--Gamma', type=float, default=0.2)   

# Pruning options.
### Note: We only use structured pruning for now. May try unstructured pruning as well unless it causes issues with CL weight sharing, but it likely shouldnt. 
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.65, help='% of neurons to prune per layer')
FLAGS.add_argument('--finetune_epochs', type=int, default=2, help='Number of epochs to finetune for after pruning')




### Data Removal Options
FLAGS.add_argument('--set_size', type=int , default=10, help='Size of sets for HSIC calculation')
FLAGS.add_argument('--num_sets', type=int , default=1, help='number os sets to be removed in HSIC')
FLAGS.add_argument('--normalize',  type=str, default='mean_std', choices=['none', 'mean_std', 'min_max'], help='which normalizing method use for hsic normalization')
FLAGS.add_argument('--layerwise', action='store_true', default=False, help='removing samples based on some layers of the network.')
FLAGS.add_argument('--setSorting', choices=['sorted', 'fixed', 'random'], default='random', help='How to order the set data prior to removal, sorted by label, shuffled, or fixed')
FLAGS.add_argument('--tau',     type=int,   default=50, help='Tau')
# Caper-specific Options
FLAGS.add_argument('--caper_epsilon',       type=float, default=0.)
FLAGS.add_argument('--Window',              type=str,   default='final')
FLAGS.add_argument('--sample_percentage',   type=float, default=0.0)
FLAGS.add_argument('--classRemovalAllowance', type=int ,  default=100)

# # HSIC Specific
# FLAGS.add_argument('--sigma', type=float, default=0., help='sigma is a hyperparameter in HSIC.')
# FLAGS.add_argument('--removed_layers',  type=str , default="layers7,8,9,10,11",  help='index of layers to be removed')
# # Energy Score Options
# FLAGS.add_argument('--T', type=float, default=1., help="Temperature for scaling the logits/activations in energy score")
# FLAGS.add_argument('--energyLayers', type=int, default=1, help='How many layers to include in energy calculation for removal')
# FLAGS.add_argument('--sortOrder', choices=['ascending', 'descending'], default='descending', help='dictates sort order for various removal methods')

# EpochAcc Options
FLAGS.add_argument('--EpochAccMetric', choices=['loss', 'softmax'], default='softmax', help='How to assess performance on training data for EpochAcc removal method')


### Generally unchanged hyperparameters
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--save_prefix', type=str, default='./checkpoints/', help='Location to save model')
FLAGS.add_argument('--steps', choices=['step2', 'step3', 'allsteps'], default='step3', help='Which steps to run')
FLAGS.add_argument('--dropout_factor', type=float, default=0.5, help='Factor for dropout layers in vgg16')





def main():
    args = FLAGS.parse_args()
   
    random.seed(args.trial_num)
    np.random.seed(args.trial_num)
    torch.manual_seed(args.trial_num)
    torch.cuda.manual_seed(args.trial_num)
    torch.cuda.manual_seed_all(args.trial_num)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    if args.sample_percentage == 0:
        args.sample_percentage = args.set_size * args.num_sets
    
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100, flush=True)   

    num_classes_by_task = utils.get_numclasses(args.dataset)
    taskid = args.task_num


    if args.dataset == "splitcifar":
        dataset = cldatasets.get_splitCIFAR(task_num=args.task_num, split = 'test', modifier=args.dataset_modifier, preprocess=args.preprocess)
    elif args.dataset == "MPC":
        dataset = cldatasets.get_mixedCIFAR_KEFMNIST(task_num=args.task_num, split = 'test', modifier=args.dataset_modifier, preprocess=args.preprocess)
    elif args.dataset == "SynthDisjoint":
        dataset = cldatasets.get_Synthetic(task_num=args.task_num, split = 'test', subset = 'disjoint', modifier=args.dataset_modifier, preprocess=args.preprocess)



    images = dataset['x']
    labels = dataset['y']



    imagesCorrupted = copy.deepcopy(images)

    if args.attack_type == "gaussian_noise":
        print("Using attack: ", args.attack_type)
        for i in range(len(imagesCorrupted)):
            imagesCorrupted[i] = corruptions.gaussian_noise(imagesCorrupted[i], severity=3)

    elif args.attack_type == "gaussian_blur":
        print("Using attack: ", args.attack_type)
        for i in range(len(imagesCorrupted)):
            imagesCorrupted[i] = corruptions.gaussian_blur(imagesCorrupted[i], severity=3)

    elif args.attack_type == "saturate":
        print("Using attack: ", args.attack_type)
        for i in range(len(imagesCorrupted)):
            imagesCorrupted[i] = corruptions.saturate(imagesCorrupted[i], severity=3)

    elif args.attack_type == "rotate":
        print("Using attack: ", args.attack_type)
        for i in range(len(imagesCorrupted)):
            imagesCorrupted[i] = corruptions.rotate(imagesCorrupted[i], severity=3)



    print("Image Corrupted size: ", imagesCorrupted.size(), flush=True)


    if args.dataset == "MPC":
        if args.task_num == 1:
            torch.save(imagesCorrupted, os.path.join(os.path.expanduser(('./data/FashionMNIST/' + str(args.task_num+1))), ('x_' + args.attack_type + '_test.bin')))

        elif args.task_num == 3:
            torch.save(imagesCorrupted, os.path.join(os.path.expanduser('./data/EMNISTL/'), ('x_' + args.attack_type + '_test.pt')))

        elif args.task_num == 5:
            torch.save(imagesCorrupted, os.path.join(os.path.expanduser('./data/KMNIST-10/' ), ('x_' + args.attack_type + '_test.pt')))

        else:
            if args.task_num == 0:
                args.task_num = 1
            torch.save(imagesCorrupted, os.path.join(os.path.expanduser(('./data/split_cifar/' + str(args.task_num))), ('x_' + args.attack_type + '_test.bin')))

    elif args.dataset == "KEF":
        torch.save(imagesCorrupted, os.path.join(os.path.expanduser(('./data/FashionMNIST/')), ('x_' + args.attack_type + '_test.bin')))

    elif args.dataset == "MPCoffset":
        torch.save(imagesCorrupted, os.path.join(os.path.expanduser(('./data/PMNIST/' + str(args.task_num))), ('x_' + args.attack_type + '_test.bin')))

    elif args.dataset == "SynthDisjoint":
        taskDict = {0:"ADM", 1:"BigGAN", 2:"Midjourney", 3:"glide", 4:"stable_diffusion_v_1_4", 5:"VQDM"}
        savepath = os.path.join(os.path.expanduser('./data/Synthetic'), 
                                taskDict[args.task_num], 'disjoint/test', args.dataset_modifier, 
                                ('X_' + args.attack_type + '.pt'))
        print("Saving to: ", savepath)
        torch.save(imagesCorrupted, savepath)



    return 0



    
    
if __name__ == '__main__':
    
    main()

