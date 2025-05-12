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
from AuxiliaryScripts.RemovalMetrics.Caper.Caper import Caper_Method
from AuxiliaryScripts.RemovalMetrics import EpochAcc



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

FLAGS.add_argument('--modifier_string', type=str, default='None,None,None,None,None,None', help='Which modifiers to use for each tasks datasets')

#!# Replaced with modifier string and list. Anywhere this was referenced, replace it with args.modifier_list[args.task_num]
# FLAGS.add_argument('--dataset_modifier', choices=['None', 'CIFAR100Full', 'OnlyCIFAR100', 'ai', 'nature'], default='None', help='Overloaded parameter for various adjustments to dataloaders in utils')
FLAGS.add_argument('--preprocess', choices=['Normalized', 'Unnormalized'], default='Unnormalized', help='Determines if the data is ranged 0:1 unnormalized or not (normalized')

FLAGS.add_argument('--attack_type', choices=['None', 'PGD', 'AutoAttack', 'gaussian_noise', 'impulse_noise', 'gaussian_blur', 'spatter', 'saturate', 'rotate'], default='PGD', help='What type of perturbation is applied')
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
FLAGS.add_argument('--set_size', type=int , default=10, help='Size of sets for removal grouping during metric calculation')
FLAGS.add_argument('--num_sets', type=int , default=1, help='number of sets to be removed in removal')
FLAGS.add_argument('--normalize',  type=str, default='mean_std', choices=['none', 'mean_std', 'min_max'], help='which normalizing method use for hsic normalization')
FLAGS.add_argument('--layerwise', action='store_true', default=False, help='removing samples based on some layers of the network.')
FLAGS.add_argument('--setSorting', choices=['sorted', 'fixed', 'random'], default='random', help='How to order the set data prior to removal, sorted by label, shuffled, or fixed')
FLAGS.add_argument('--tau',     type=int,   default=50, help='Tau')
# Caper-specific Options
FLAGS.add_argument('--caper_epsilon',       type=float, default=0.)
FLAGS.add_argument('--Window',              type=str,   default='final')
FLAGS.add_argument('--sample_percentage',   type=float, default=0.0)
FLAGS.add_argument('--classRemovalAllowance', type=int ,  default=100)

### Leaving these in just in case removing them breaks something, to add context as to what they were doing. This measure did not end up being used for the paper as it was not effective.
# # Energy Score Options
# FLAGS.add_argument('--T', type=float, default=1., help="Temperature for scaling the logits/activations in energy score")
# FLAGS.add_argument('--energyLayers', type=int, default=1, help='How many layers to include in energy calculation for removal')
# FLAGS.add_argument('--sortOrder', choices=['ascending', 'descending'], default='descending', help='dictates sort order for various removal methods')

# EpochAcc Options
FLAGS.add_argument('--EpochAccMetric', choices=['loss', 'softmax'], default='softmax', help='How to assess performance on training data for EpochAcc removal method')
FLAGS.add_argument('--EpochAccEpochs', type=int, default=-1, help='How many epochs to consider when calculating metric')
FLAGS.add_argument('--EpochAccInterval', type=int, default=1, help='Consider metric averaged for every Nth epoch')


### Generally unchanged hyperparameters
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--save_prefix', type=str, default='./checkpoints/', help='Location to save model')
FLAGS.add_argument('--steps', choices=['step1', 'step2', 'step3', 'allsteps'], default='step3', help='Which steps to run')
FLAGS.add_argument('--dropout_factor', type=float, default=0.5, help='Factor for dropout layers in vgg16')




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

    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100, flush=True)   

    num_classes_by_task = utils.get_numclasses(args.dataset, modifier=args.modifier_list[args.task_num])
    taskid = args.task_num


    ###################
    ##### Prepare Checkpoint and Manager
    ###################
    ### Load the previous checkpoint if there is one and set the save path
    args.save_prefix, loadpath = utils.load_task_paths(args)
    loadpath, ckpt = utils.load_task_checkpoint(args, loadpath)

    manager = Manager(args, ckpt, first_task_classnum=num_classes_by_task[taskid])
    
    if args.pretrained and args.task_num==0:
        #*# Load a compatible version of the pretrained weights
        pretrained_dict = utils.load_pretrained(args, manager)
        manager.network.model.load_state_dict(pretrained_dict, strict=False)







    ###################
    ##### Setup task and data
    ###################

    ### Logic for looping over remaining tasks
    taskid = args.task_num
    
    print("Task ID: ", taskid, " #", args.task_num, " in sequence for dataset: ", args.dataset)
    print('\n\n args.save_prefix  is ', args.save_prefix, "\n\n", flush=True)
    os.makedirs(args.save_prefix, exist_ok = True)

    manager.save_prefix = args.save_prefix

    trained_path, finetuned_path = os.path.join(args.save_prefix, "trained.pt"), os.path.join(args.save_prefix, "final.pt") 
    print("Finetuned path: ", finetuned_path, flush=True)

    ### Prepare dataloaders for new task
    ### Note: For some datasets we just use the test data for the val dataset or vice versa
    train_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="train", preprocess=args.preprocess, shuffle=True, modifier=args.modifier_list[args.task_num])
    val_data_loader  =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="valid", preprocess=args.preprocess, modifier=args.modifier_list[args.task_num])
    extra_data_loader =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="train", preprocess=args.preprocess, modifier=args.modifier_list[args.task_num])
    manager.train_loader, manager.val_loader, manager.extra_loader = train_data_loader, val_data_loader, extra_data_loader

    total_images = sum(len(batch[0]) for batch in manager.extra_loader)
    print('\n\n number of extra samples :', total_images, flush=True)

    if args.dataset == "splitcifar":
        dataset = cldatasets.get_splitCIFAR(task_num=args.task_num, split = 'train', preprocess=args.preprocess, modifier=args.modifier_list[args.task_num])
    elif args.dataset == "MPC":
        dataset = cldatasets.get_mixedCIFAR_KEFMNIST(task_num=args.task_num, split = 'train', preprocess=args.preprocess, modifier=args.modifier_list[args.task_num])
    elif args.dataset == "SynthDisjoint":
        dataset = cldatasets.get_Synthetic(task_num=args.task_num, split = 'train', modifier=args.modifier_list[args.task_num], preprocess=args.preprocess)
    elif args.dataset in ["ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"]:
        dataset = cldatasets.get_Synthetic_SingleGenerator(task_num=args.task_num, split = 'train', generator = args.dataset, modifier=args.modifier_list[args.task_num], preprocess=args.preprocess)

    ### Initialize the z (unique sample ID) values of the dataset before changing anything
    dataset['z'] = torch.arange(len(dataset['y']))

    #*# Previously the dataset passed in would be shuffled, affecting the original copy. Fixed by passing in a copy
    ### Note: Because the shuffling mode was restricted to not changing the order of samples, this didn't actually break anything up to this point
    all_batches = utils.prepare_allbatches(set_size=args.set_size, dataset=copy.deepcopy(dataset))
    manager.hsic_dataset = all_batches
    

    ### This is the standard setup used for experiments to dictate percentage removal. Its a little roundabout and will be streamlined in the future
    if args.sample_percentage < 0.01:
        total_images = sum(len(batch[0]) for batch in manager.train_loader)
        print('\n\n number of train samples :', total_images, flush=True)
        args.num_sets = round((args.num_sets * total_images)/100)
        args.sample_percentage = args.set_size * args.num_sets
        args.classRemovalAllowance = floor(args.sample_percentage / num_classes_by_task[taskid])

        print("\nUpdated arguments:")
        print('args.num_sets is:', args.num_sets)
        print('args.sample_percentage is:', args.sample_percentage)
        print('args.classRemovalAllowance is:', args.classRemovalAllowance)
    


    ### This is for producing and setting the classifier layer for a given task's # classes
    manager.network.add_dataset(str(args.task_num), num_classes_by_task[taskid])
    manager.network.set_dataset(str(args.task_num))
    if args.cuda:
        manager.network.model = manager.network.model.cuda()





    ### Track the softmax of normal and attacked samples at tau, the logits of epochs in step 1, and whether the sample's removed
    sampledict = {"tau_logits":{}, "tau_advlogits":{}, "epochlogits": {}, "removed":{}, "labels":{}, 'step2_time':{}}
    for i in range(len(dataset['y'])):
        sampledict['tau_logits'][i] = []
        sampledict['tau_advlogits'][i] = []
        sampledict['epochlogits'][i] = []
        sampledict['removed'][i] = 0
        sampledict['labels'][i] = 0
        sampledict['step2_time'][i] = 0

    ### Passing this to manager so that the step 1 logits can be passed back
    manager.sampledict = sampledict


    
    ### Reload all previously masked weights to get the full network prior to weight sharing.
    if args.task_num != 0:
        manager.prepare_task()

    ### Changed to copy the manager after all the setup steps have been done to avoid needing to repeat them
    manager_deep_copy = copy.deepcopy(manager)





    ##################################################################################################################
    ##### Step 1
    #####    - train non-adversarially for tau epochs
    ##################################################################################################################
    if args.removal_metric != "NoRemoval" and args.tau != 0:
        print('\n\n\n', '-' * 16, '\nstep 1 is started \n', flush=True)

        trained_path = os.path.join(args.save_prefix, (args.removal_metric + "1-2_trained.pt"))
        trained_path_layerwise = os.path.join(args.save_prefix, (args.removal_metric + "1-2_trained-layerwise.pt"))
        
        manager.train(args.tau, save=False, savename=trained_path, num_class=num_classes_by_task[taskid], use_attack=False, save_best_model=True, trackpreds=True)


        ### Store the logits for each epoch of step 1 in the dict
        sampledict['epochlogits'] = manager.sampledict['epochlogits']
        sampledict['labels'] = manager.sampledict['labels']


        ##################################################################################################################
        ##### Step 2
        #####    - remove samples predicted to reduce robustness/accuracy of model based on chosen metric
        ##################################################################################################################

        print("\n\n\n", '-' * 16, '\nstep 2 is started \n', flush=True)
        print("Removing data with metric: ", args.removal_metric, '\n\n')

        if args.pretrained == True:
            pretrain_string = "pretrained"
        else:
            pretrain_string = "not_pretrained"

        

    if args.removal_metric not in ["NoRemoval", 'step1']:

        if args.removal_metric == "Random":

            print("\n\nCalculating random")
            #!# Remove data and return a new train loader based on the chosen removal metric
            ##continue training with new dataset driven from removing some sets
            ### For sample percentage setting, args.num_sets was reassigned the value of the percent to remove already
            train_new_data_loader_random, sets_to_remove_random = utils.random_remove(all_batches, args.num_sets, args.batch_size)
            sets_to_remove_random = torch.tensor(sets_to_remove_random)
            torch.save(sets_to_remove_random, (args.save_prefix + "/removed_indices_random.pt"))
        
            train_new_data_loader = train_new_data_loader_random
        


            ### Not which samples were removed
            for i in range(len(sets_to_remove_random)):
                IDs = all_batches[sets_to_remove_random[i]][2]
                for ID in IDs:
                    sampledict['removed'][ID.item()] = 1




        elif args.removal_metric == "EpochAcc":
            EpochAccClass = EpochAcc.EpochAcc_Method(args, manager.network.model, extra_data_loader, sampledict=sampledict)
            print("\n\nCalculating EpochAcc")
            EpochAcc_mask = EpochAccClass.gen_data_mask()
            ### The mask generation is derived from caper's setup so we're reusing the removal function provided the same type of mask
            train_new_data_loader_EpochAcc = utils.caper_remove(dataset, EpochAcc_mask, args.batch_size)
            sets_to_remove_EpochAcc = torch.from_numpy(EpochAcc_mask)
            print("Sets to remove EpochAcc: ", sets_to_remove_EpochAcc)
            torch.save(sets_to_remove_EpochAcc, (args.save_prefix + "/removed_indices_EpochAcc.pt"))

            train_new_data_loader = train_new_data_loader_EpochAcc

            sampledict = EpochAccClass.sampledict



        elif args.removal_metric == "Caper":
            CaperClass = Caper_Method(args, manager.network.model, extra_data_loader, sampledict=sampledict)
            print("\n\nCalculating caper")
            caper_mask = CaperClass.New_Data(args.save_prefix)
            train_new_data_loader_caper = utils.caper_remove(dataset, caper_mask, args.batch_size)
            sets_to_remove_caper = torch.from_numpy(caper_mask)
            print("Sets to remove caper: ", sets_to_remove_caper)
            torch.save(sets_to_remove_caper, (args.save_prefix + "/removed_indices_caper.pt"))

            train_new_data_loader = train_new_data_loader_caper

            sampledict = CaperClass.sampledict




        sampledict_path = os.path.join(args.save_prefix, ("sampledict.pt"))
        torch.save(sampledict, sampledict_path)


        manager.train_loader = train_new_data_loader

        # trained_path = os.path.join(args.save_prefix, ((args.removal_metric + "2-trained.pt")))
        # utils.save_ckpt(manager, savename=trained_path)


        # manager.network.check_weights()

        
        total_sum = 0
        labelcount = {}
        for _, labels, _ in train_new_data_loader:
            for label in labels:
                if label.item() in labelcount.keys():
                    labelcount[label.item()] += 1
                else:
                    labelcount[label.item()] = 1
                total_sum += 1
            # total_sum += torch.sum(labels).item()
        print(labelcount)
        print('all_batches sum of lables is', total_sum)
        
        sorted_by_keys = {key: labelcount[key] for key in sorted(labelcount)}
        print(sorted_by_keys)












        ##################################################################################################################
        ##### Step 3
        #####   - Adversarially train to convergence and check test accuracy afterwards. Prune and Finetune if continuing to next task
        ##################################################################################################################
        


        ### Reloading checkpoint stored at start of task
        manager = manager_deep_copy
        manager.train_loader = train_new_data_loader
        


        # manager.network.check_weights()
            
        num_samples = len(manager.train_loader.dataset)
        print(f"Number of samples in the new train dataLoader for step 3: {num_samples}")



    #*# Run step 3 if set to do so, otherwise we skip it if just interested in reporting metrics at epoch tau
    if args.steps in ["step3", 'allsteps']:
        print("\n\n\n", '-' * 16, '\nstep 3 is started \n', flush=True)

        if args.cuda:
            manager.network.model = manager.network.model.cuda()
            
        trained_path = os.path.join(args.save_prefix, ((args.removal_metric + "steps3-trained.pt")))
        manager.train(args.train_epochs, save=True, savename=trained_path, num_class=num_classes_by_task[taskid])
        utils.save_ckpt(manager, savename=trained_path)


        if args.attack_type in ['PGD', 'AutoAttack', 'None']:
            print("Getting test accuracies for attack type: ", args.attack_type)
            test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, modifier=args.modifier_list[args.task_num])

            test_errors, test_errors_attacked = manager.eval(num_classes_by_task[taskid],  use_attack=True, Data=test_data_loader)
            test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
            test_accuracy_attacked = 100 - test_errors_attacked[0]  # Top-1 accuracy.

            accsPrint = [test_accuracy, test_accuracy_attacked]
            print('Final Test Vanilla and Attacked Accuracy: ', accsPrint)




        elif args.attack_type in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:

            accsList = []
            print("Getting test accuracies for normal test data")
            test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, modifier=args.modifier_list[args.task_num])

            test_errors = manager.eval(num_classes_by_task[taskid],  use_attack=False, Data=test_data_loader)
            test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
            print('Final Test Vanilla Accuracy on un-attacked data: %0.2f%%' %(test_accuracy))
            accsList.append(test_accuracy)

            for attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:
                print("\nGetting test accuracies for corrupted test data with corruption: ", attack)
                test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, attack_type=attack, modifier=args.modifier_list[args.task_num])

                test_errors = manager.eval(num_classes_by_task[taskid],  use_attack=False, Data=test_data_loader)
                test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
                print('Final Test Accuracy on ', attack ,' data: %0.2f%%' %(test_accuracy))
      
                accsList.append(test_accuracy)

            print("All accs: ", accsList)



    ### If continuing to next task, then also do pruning and finetuning
    if args.steps in ['allsteps']:

        ### Prune unecessary weights or nodes
        manager.prune()
        print('\nPost-prune eval:')


        if args.attack_type in ['PGD', 'AutoAttack', 'None']:
            test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, modifier=args.modifier_list[args.task_num])

            test_errors, test_errors_attacked = manager.eval(num_classes_by_task[taskid],  use_attack=True, Data=test_data_loader)
            test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
            test_accuracy_attacked = 100 - test_errors_attacked[0]  # Top-1 accuracy.

            print('Pruned Test Adversarial Accuracy on attacked data: %0.2f%%' %(test_accuracy_attacked))  
            print('Pruned Test Vanilla Accuracy on un-attacked data: %0.2f%%' %(test_accuracy))


        elif args.attack_type in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:

            accsList = {}
            test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, modifier=args.modifier_list[args.task_num])
            test_errors = manager.eval(num_classes_by_task[taskid],  use_attack=False, Data=test_data_loader)
            test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.

            accsList['Normal'] = test_accuracy

            for attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:
                test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, attack_type=attack, modifier=args.modifier_list[args.task_num])
                test_errors = manager.eval(num_classes_by_task[taskid],  use_attack=False, Data=test_data_loader)
                test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.      

                accsList[attack] = test_accuracy

            print("All accs: ", accsList)
            print("All accs values: ", list(accsList.values()))


        utils.save_ckpt(manager, finetuned_path)


        if args.finetune_epochs:
            print('Doing some extra finetuning...')
            manager.train(args.finetune_epochs, save=True, savename=finetuned_path, num_class=num_classes_by_task[taskid])

        ### Save the checkpoint and move on to the next task if required
        utils.save_ckpt(manager, finetuned_path)



        ### Get test accuracies after finetuning

        if args.attack_type in ['PGD', 'AutoAttack', 'None']:
            print("Getting test accuracies for attack type: ", args.attack_type)
            test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, modifier=args.modifier_list[args.task_num])

            test_errors, test_errors_attacked = manager.eval(num_classes_by_task[taskid],  use_attack=True, Data=test_data_loader)
            test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
            test_accuracy_attacked = 100 - test_errors_attacked[0]  # Top-1 accuracy.

            accsPrint = [test_accuracy, test_accuracy_attacked]
            print('Final Test Vanilla and Attacked Accuracy: ', accsPrint)



        elif args.attack_type in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:

            accsList = []
            print("Getting test accuracies for normal test data")
            test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, modifier=args.modifier_list[args.task_num])

            test_errors = manager.eval(num_classes_by_task[taskid],  use_attack=False, Data=test_data_loader)
            test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
            print('Finetuned Final Test Vanilla Accuracy on un-attacked data: %0.2f%%' %(test_accuracy))
            accsList.append(test_accuracy)

            for attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:
                print("\nGetting test accuracies for corrupted test data with corruption: ", attack)
                test_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test", preprocess=args.preprocess, attack_type=attack, modifier=args.modifier_list[args.task_num])

                test_errors = manager.eval(num_classes_by_task[taskid],  use_attack=False, Data=test_data_loader)
                test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
                print('Finetuned Final Test Accuracy on ', attack ,' data: %0.2f%%' %(test_accuracy))
      
                accsList.append(test_accuracy)

            print("All accs: ", accsList)




        print('-' * 16)
        print('Pruning summary:')
        manager.network.check(True)
        print('-' * 16)
        print("\n\n\n\n")





    return 0



    
    
if __name__ == '__main__':
    
    main()

