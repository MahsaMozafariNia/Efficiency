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
import random
import time

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.optim.lr_scheduler  import MultiStepLR
import numpy as np
import pandas as pd

from itertools import islice
from math import floor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

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

FLAGS.add_argument('--dataset', type=str, choices=['MPC', 'splitcifar', 'SynthDisjoint',
                                                "ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"], default='splitcifar', help='Name of dataset')


#!# Replaced with modifier string and list. Anywhere this was referenced, replace it with args.modifier_list[args.task_num]
# FLAGS.add_argument('--dataset_modifier', choices=['None', 'CIFAR100Full', 'OnlyCIFAR100', 'ai', 'nature'], default='None', help='Overloaded parameter for various adjustments to dataloaders in utils')
FLAGS.add_argument('--preprocess', choices=['Normalized', 'Unnormalized'], default='Unnormalized', help='Determines if the data is ranged 0:1 unnormalized or not (normalized')

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
# HSIC Specific
FLAGS.add_argument('--sigma', type=float, default=2., help='sigma is a hyperparameter in HSIC.')
FLAGS.add_argument('--removed_layers',  type=str , default="layers7,8,9,10,11",  help='index of layers to be removed')
# Caper-specific Options
FLAGS.add_argument('--caper_epsilon',       type=float, default=0.)
FLAGS.add_argument('--Window',              type=str,   default='final')
FLAGS.add_argument('--sample_percentage',   type=float, default=0.0)
FLAGS.add_argument('--classRemovalAllowance', type=int ,  default=100)
# Energy Score Options
FLAGS.add_argument('--T', type=float, default=1., help="Temperature for scaling the logits/activations in energy score")
FLAGS.add_argument('--energyLayers', type=int, default=1, help='How many layers to include in energy calculation for removal')
FLAGS.add_argument('--sortOrder', choices=['ascending', 'descending'], default='descending', help='dictates sort order for various removal methods')
# EpochAcc Options
FLAGS.add_argument('--EpochAccMetric', choices=['loss', 'softmax'], default='softmax', help='How to assess performance on training data for EpochAcc removal method')
FLAGS.add_argument('--EpochAccEpochs', type=int, default=-1, help='How many epochs to consider when calculating metric')
FLAGS.add_argument('--EpochAccInterval', type=int, default=1, help='Consider metric averaged for every Nth epoch')


### Generally unchanged hyperparameters
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--save_prefix', type=str, default='./checkpoints/', help='Location to save model')
FLAGS.add_argument('--steps', choices=['step1', 'step2', 'step3', 'allsteps'], default='step3', help='Which steps to run')
FLAGS.add_argument('--dropout_factor', type=float, default=0.5, help='Factor for dropout layers in vgg16')


### Arguments dictating what checkpoint is loaded
FLAGS.add_argument('--attack_type', choices=['None', 'PGD', 'gaussian_noise', 'impulse_noise', 'gaussian_blur', 'spatter', 'saturate', 'rotate'], default='PGD', help='What type of perturbation is applied')
FLAGS.add_argument('--modifier_string', type=str, default='None,None,None,None,None,None', help='Which modifiers to use for each tasks datasets')

### Arguments dictating what task setup we are evaluating on
FLAGS.add_argument('--eval_tasknum', type=int, default=-1, help='Which task to evaluate on (should be <= args.task_num)')
FLAGS.add_argument('--eval_modifier', type=str, default='None', help='Which modifiers to use for each tasks datasets')
FLAGS.add_argument('--eval_attack_type', choices=['None', 'PGD', 'AutoAttack', 'gaussian_noise', 'impulse_noise', 'gaussian_blur', 'spatter', 'saturate', 'rotate'], default='PGD', help='What type of perturbation is applied')






def load_task_paths(args=None):
    if args == None:
        print("!!!No arguments provided for loading checkpoint in utils.load_task_checkpoint")
        return "", None


    if args.use_train_scheduler==True:
        loadpath = os.path.join("./checkpoints/", str(args.dataset) + "_" + str(args.arch), str(args.run_id), 'trial-'+ str(args.trial_num),
                                    str(args.attack_type), str(args.prune_perc_per_layer),
                                    'epochs-'+ str(args.train_epochs) + "_batch_size-" + str(args.batch_size), 'using_scheduler')
    else:
        loadpath= os.path.join("./checkpoints/", str(args.dataset) + "_" + str(args.arch), str(args.run_id), 'trial-'+ str(args.trial_num),
                                    str(args.attack_type), str(args.prune_perc_per_layer),
                                    'epochs-'+ str(args.train_epochs) + "_batch_size-" + str(args.batch_size), 'lr-'+str(args.lr),
                                    'patience-'+str(args.lr_patience)+'_'+'factor-'+str(args.lr_factor)+'_'+ 'lrmin-'+str(args.lr_min))
    
    savepath = loadpath


    ### All save paths start in a directory based on the shared hyperparameter values for that metric
    if args.removal_metric in ['Caper']:
        savepath = os.path.join(savepath, '_tau-'+ str(args.tau),  'metric-' + str(args.removal_metric) + '_normalize-' + str(args.normalize), 
                                'num_sets-'+ str(args.num_sets) + '_set_size-'+ str(args.set_size) + "_setSorting-" + args.setSorting)             
    elif args.removal_metric in ['EpochAcc']:
        savepath = os.path.join(savepath, '_tau-'+ str(args.tau),  'metric-' + str(args.removal_metric) + '_normalize-' + str(args.normalize)+ '_sortOrder-' + str(args.sortOrder), 'EpochAccMetric-'+ str(args.EpochAccMetric), 
                                'NumEpochs-' + str(args.EpochAccEpochs) + '_Interval-' + str(args.EpochAccInterval), 'num_sets-'+ str(args.num_sets) + '_set_size-'+ str(args.set_size) + "_setSorting-" + args.setSorting) 
    elif args.removal_metric in ['Random']:
        savepath = os.path.join(savepath, '_tau-'+ str(args.tau),  'metric-' + str(args.removal_metric) + '_normalize-' + str(args.normalize), 
                                'num_sets-'+ str(args.num_sets) + '_set_size-'+ str(args.set_size) + "_setSorting-" + args.setSorting) 

    elif args.removal_metric != 'NoRemoval':
        savepath = os.path.join(savepath, '_tau-'+ str(args.tau),  'metric-' + str(args.removal_metric) + '_normalize-' + str(args.normalize)+ '_sortOrder-' + str(args.sortOrder), 
                                'num_sets-'+ str(args.num_sets) + '_set_size-'+ str(args.set_size) + "_setSorting-" + args.setSorting) 









    #!# This is a VERY clunky setup to avoid rerunning experiments. When not using a modifier, we dont load from a nested path
    #!#    but when using one (synthetic dataset) we will save/load baselines in a nested manner to be compatible with removal setups
    if args.load_from == "baseline" and args.modifier_list[0] == "None":
        ### NoRemoval file is shared between different tau values, but removal metrics are not
        loadpath = os.path.join(loadpath, (str(args.task_num-1)+"_NoRemoval"))

        ### No nesting of paths done for MPC (when modifier ~= "ai" or "nature") when loading from baseline
        savepath = os.path.join(savepath, (str(args.task_num) + "_" + args.removal_metric))   
        # pass
    ### If loading from removal metrics or using synthetic datasets' "ai" and "nature" modifiers
    else:
        ### Load from within the nested removal directories
        #!# If loading from removal metric, need the loadpath to reflect the appropriate hyperparameter subdirectories
        if args.load_from != "baseline":
            loadpath = savepath



        ### Update loadpath for each previous task in the current sequence
        for t in range(0, args.task_num):
            ### Setup the appropriate modifier string for the given task. If not doing synthetic data, set up to be empty for compatibility
            ###     with old saves
            if args.modifier_list[t] == "None":
                task_modifier = ""
            else:
                task_modifier = (args.modifier_list[t] + "_")

            if args.load_from == "baseline":
                loadpath = os.path.join(loadpath, (task_modifier + str(t) + "_" + "NoRemoval"))
                savepath = os.path.join(savepath, (task_modifier + str(t) + "_" + "NoRemoval"))
            else:
                ### Note: baseline is saved as "NoRemoval" metric
                loadpath = os.path.join(loadpath, (task_modifier + str(t) + "_" + args.removal_metric))
                savepath = os.path.join(savepath, (task_modifier + str(t) + "_" + args.removal_metric))


        ### Extend the savepath from the loadpath to include the current task
        if args.modifier_list[args.task_num] == "None":
            task_modifier = ""
        else:
            task_modifier = (args.modifier_list[args.task_num] + "_")


        ### Note: baseline is saved as "NoRemoval" metric
        savepath = os.path.join(savepath, (task_modifier + str(args.task_num) + "_" + args.removal_metric))
        loadpath = os.path.join(loadpath, (task_modifier + str(args.task_num) + "_" + args.removal_metric))

    return savepath, loadpath




def load_task_checkpoint(args=None, loadpath=None):
    ckpt = None

    ### If no checkpoint is found, the default value will be None and a new one will be initialized in the Manager
    ### Path to load previous task's checkpoint, if not starting at task 0
    ### Since baseline was designed to report accuracies on the terminal task we use the unpruned, terminal network
    if args.load_from == "baseline":
        previous_task_path = os.path.join(loadpath, (args.removal_metric + "steps3-trained.pt")) 
    ### Otherwise we use the prunnd and finetuned network since these are how accuracies were reported for 6-task sequences
    else:
        previous_task_path = os.path.join(loadpath, "final.pt") 
    print('path is', previous_task_path)
    ### Reloads checkpoint depending on where you are at for the current task's progress (t->c->p)    
    if os.path.isfile(previous_task_path) == True:
        ckpt = torch.load(previous_task_path)
        print("Checkpoint found and loaded from: ", previous_task_path)
    else:
        print("!!!No checkpoint file found at ", previous_task_path)

    return loadpath, ckpt








 






#***# Save structure: Runid is the experiment, different task orders are subdirs that share up to the last common task so that the task can be reused/located just by giving the runid and task sequence and can be shared between multiple alternative orders for efficiency
###    Basically this just means 6 nested directories, which are nested in order of task order for the given experiment. So all subdirs of the outermost directory 2 have task 2 as the first task and can share the final dict from task 2 amongst eachother for consistency and efficiency
def main():
    args = FLAGS.parse_args()
   

    args.modifier_list = args.modifier_string.split(',')
    random.seed(args.trial_num)
    np.random.seed(args.trial_num)
    torch.manual_seed(args.trial_num)
    torch.cuda.manual_seed(args.trial_num)
    torch.cuda.manual_seed_all(args.trial_num)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    if args.eval_tasknum == -1:
        args.eval_tasknum = args.task_num

    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100, flush=True)   

    num_classes_by_task = utils.get_numclasses(args.dataset, modifier=args.modifier_list[args.task_num])
    taskid = args.eval_tasknum


    ###################
    ##### Prepare Checkpoint and Manager
    ###################
    ### Load the previous checkpoint if there is one and set the save path
    args.save_prefix, loadpath = load_task_paths(args)


    loadpath, ckpt = load_task_checkpoint(args, loadpath)


    manager = Manager(args, ckpt, first_task_classnum=num_classes_by_task[taskid])
    
    manager.task_num = args.eval_tasknum






    ###################
    ##### Setup task and data
    ###################



    ### This is for producing and setting the classifier layer for a given task's # classes
    manager.network.set_dataset(str(taskid))

    
    if args.cuda:
        manager.network.model = manager.network.model.cuda()




    ### Now that the loading is done, change the attack type to the one used for evaluation
    args.attack_type = args.eval_attack_type
    manager.args.attack_type = args.eval_attack_type



    if args.attack_type in ['PGD', 'AutoAttack', 'None']:
        test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, modifier=args.eval_modifier)

        test_errors, test_errors_attacked = manager.eval(num_classes_by_task[taskid],  use_attack=True, Data=test_data_loader)
        test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
        test_accuracy_attacked = 100 - test_errors_attacked[0]  # Top-1 accuracy.


        accsPrint = [test_accuracy, test_accuracy_attacked]
        print('Final Test Vanilla and Attacked Accuracy: ', accsPrint)

    elif args.attack_type in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:

        accsList = {}
        test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, modifier=args.eval_modifier)
        test_errors = manager.eval(num_classes_by_task[taskid],  use_attack=False, Data=test_data_loader)
        test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.

        accsList['Normal'] = test_accuracy

        for attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:
            test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, attack_type=attack, modifier=args.eval_modifier)
            test_errors = manager.eval(num_classes_by_task[taskid],  use_attack=False, Data=test_data_loader)
            test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.      

            accsList[attack] = test_accuracy

        print("All accs: ", accsList)
        print("All accs values: ", list(accsList.values()))







    return 0



    
    
if __name__ == '__main__':
    
    main()

