"""Contains utility functions for calculating activations and connectivity. Adapted code is acknowledged in comments"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from AuxiliaryScripts import DataGenerator as DG
from AuxiliaryScripts import cldatasets
from AuxiliaryScripts import network as net
import torch.utils.data as D
import time
import copy
import math
import sklearn
import random 

import scipy.spatial     as ss

from math                 import log, sqrt
from scipy                import stats
from sklearn              import manifold
from scipy.special        import *
from sklearn.neighbors    import NearestNeighbors





#####################################################
###    Activation Functions
#####################################################






acts = {}

### Returns a hook function directed to store activations in a given dictionary key "name"
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        acts[name] = output.detach().cpu()
    return hook

### Create forward hooks to all layers which will collect activation state
### Collected from ReLu layers when possible, but not all resnet18 trainable layers have coupled relu layers
def get_all_layers(net, hook_handles, relu_idxs):
    for module_idx, (name,module) in enumerate(net.named_modules()):
        if module_idx in relu_idxs:
            hook_handles.append(module.register_forward_hook(getActivation(module_idx)))


### Process and record all of the activations for the given pair of layers
def activations(data_loader, model, cuda, act_idxs, use_relu = False, sampledict=None, attacked=False):
    temp_op       = None
    temp_label_op = None

    parents_op  = None
    labels_op   = None

    handles     = []

    ### Set hooks in all tunable layers
    
    get_all_layers(model, handles, act_idxs)

    ### A dictionary for storing the activations
    actsdict = {}
    labels = None

    for i in act_idxs: 
        actsdict[i] = None
    
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label, IDs = data
            out = model(x_input.cuda())


            if sampledict:
                if attacked:
                    for i, ID in enumerate(IDs):
                        sampledict['tau_advlogits'][ID.item()].append(out[i]) 
                else:
                    for i, ID in enumerate(IDs):
                        sampledict['tau_logits'][ID.item()].append(out[i]) 



            if step == 0:
                labels = y_label.detach().cpu()
                for key in acts.keys():
                    
                    ### We need to convert from relu idxs to trainable layer idxs for future masking purposes

                    ### For all conv layers we average over the feature maps, this makes them compatible when comparing with linear layers and reduces memory requirements
                    if use_relu:
                        acts[key] = F.relu(acts[key])
                        
                    if len(acts[key].shape) > 2:
                        actsdict[key] = acts[key].mean(dim=3).mean(dim=2)
                    else:
                        actsdict[key] = acts[key]
            else: 
                labels = torch.cat((labels, y_label.detach().cpu()),dim=0)
                for key in acts.keys():
                    if use_relu:
                        acts[key] = F.relu(acts[key])

                    if len(acts[key].shape) > 2:
                        actsdict[key] = torch.cat((actsdict[key], acts[key].mean(dim=3).mean(dim=2)), dim=0)
                    else:
                        actsdict[key] = torch.cat((actsdict[key], acts[key]), dim=0)

            
    # Remove all hook handles
    for handle in handles:
        handle.remove()    

    return actsdict, labels, sampledict




#####################################################
###    Saving Function
#####################################################



### Saves a checkpoint of the model
def save_ckpt(manager, savename):
    """Saves model to file."""

    # Prepare the ckpt.
    ckpt = {
        'args': manager.args,
        'all_task_masks': manager.all_task_masks,
        'network': manager.network,
    }

    print("Saving checkpoint to ", savename)
    # Save to file.
    torch.save(ckpt, savename)





#####################################################
###    Masking Functions
#####################################################

### Get a binary mask where all previously frozen weights are indicated by a value of 1
### After pruning on the current task, this will still return the same masks, as the new weights aren't frozen until the task ends
def get_frozen_mask(weights, module_idx, all_task_masks, task_num):
    mask = torch.zeros(weights.shape)
    ### Include all weights used in past tasks (which would have been subsequently frozen)
    for i in range(0, task_num):
        if i == 0:
            mask = all_task_masks[i][module_idx].clone().detach()
        else:
            mask = torch.maximum(all_task_masks[i][module_idx], mask)
    return mask
        
    
### Get a binary mask where all unpruned, unfrozen weights are indicated by a value of 1
### Unlike get_frozen_mask(), this mask will change after pruning since the pruned weights are no longer trainable for the current task
def get_trainable_mask(module_idx, all_task_masks, task_num):
    mask = all_task_masks[task_num][module_idx].clone().detach()
    frozen_mask = get_frozen_mask(mask, module_idx, all_task_masks, task_num)
    mask[frozen_mask.eq(1)] = 0
    return mask
    

        




#####################################################
###    Dataset Functions
#####################################################


### Number of classes by task
def get_numclasses(dataset, modifier=None):
    if dataset == 'MPC':
        if modifier == "CIFAR100Full":
            numclasses = [100,10,20,10,20,10]
        else:
            numclasses = [20,10,20,26,20,10]
    elif dataset == 'splitcifar':
        if modifier == "OnlyCIFAR100":
            numclasses = [20,20,20,20,20]
        else:
            numclasses = [10,20,20,20,20,20]
    elif dataset in ['SynthDisjoint', "ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"]:
        numclasses = [100,100,100,100,100,100]
    
    return numclasses
    
### Returns a dictionary of "train", "valid", and "test" data+labels for the appropriate cifar subset
def get_dataloader(dataset, batch_size, num_workers=4, pin_memory=False, normalize=None, task_num=0, set="train", preprocess="Normalized", shuffle=False, attack_type=None, modifier=None):

    # standard split CIFAR-10/100 sequence of tasks

    if dataset == "MPC":
        dataset = cldatasets.get_mixedCIFAR_KEFMNIST(task_num=task_num, split = set, preprocess=preprocess, attack=attack_type, modifier=modifier)
    elif dataset == "splitcifar":
        dataset = cldatasets.get_splitCIFAR(task_num=task_num, split = set, preprocess=preprocess, attack=attack_type, modifier=modifier)
    elif dataset == "SynthDisjoint":
        dataset = cldatasets.get_Synthetic(task_num=task_num, split = set, modifier=modifier, preprocess=preprocess, attack=attack_type)


    elif dataset in ["ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"]:
        dataset = cldatasets.get_Synthetic_SingleGenerator(task_num=task_num, split = set, generator = dataset, modifier=modifier, preprocess=preprocess, attack=attack_type)



    else: 
        print("Incorrect dataset for get_dataloader()")
        return -1
        

    
    IDs = torch.arange(len(dataset['y']))
    # print("Size of IDs: ", IDs.size(), "min max: ", IDs.min(), " ", IDs.max())
    ### Makes a custom dataset for a given dataset through torch
    # generator = DG.SimpleDataGenerator(dataset['x'],dataset['y'])
    generator = DG.IdTrackDataGenerator(dataset['x'],dataset['y'], IDs)
    
    


    
    ### Loads the custom data into the dataloader
    if set == "train":        
        return data.DataLoader(generator, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory=pin_memory)
    else:
        return data.DataLoader(generator, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory=pin_memory)












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
    if args.removal_metric in ['HSIC']:
        savepath = os.path.join(savepath, '_tau-'+ str(args.tau), 'metric-' + str(args.removal_metric) + '_normalize-' + str(args.normalize), 'tau-'+ str(args.tau), 
                                'num_sets-'+ str(args.num_sets) + '_set_size-'+ str(args.set_size) + "_setSorting-" + args.setSorting)     
    elif args.removal_metric in ['Caper']:
        savepath = os.path.join(savepath, '_tau-'+ str(args.tau),  'metric-' + str(args.removal_metric) + '_normalize-' + str(args.normalize), 
                                'num_sets-'+ str(args.num_sets) + '_set_size-'+ str(args.set_size) + "_setSorting-" + args.setSorting)             
    elif args.removal_metric in ['EnergyFirst', 'EnergyLast', 'Energy']:
        savepath = os.path.join(savepath, '_tau-'+ str(args.tau),  'metric-' + str(args.removal_metric) + '_normalize-' + str(args.normalize)+ '_sortOrder-' + str(args.sortOrder), 'energyLayers-'+ str(args.energyLayers), 
                                'num_sets-'+ str(args.num_sets) + '_set_size-'+ str(args.set_size) + "_setSorting-" + args.setSorting) 
    elif args.removal_metric in ['EpochAcc']:
        savepath = os.path.join(savepath, '_tau-'+ str(args.tau),  'metric-' + str(args.removal_metric) + '_normalize-' + str(args.normalize)+ '_sortOrder-' + str(args.sortOrder), 'EpochAccMetric-'+ str(args.EpochAccMetric), 
                                'NumEpochs-' + str(args.EpochAccEpochs) + '_Interval-' + str(args.EpochAccInterval), 'num_sets-'+ str(args.num_sets) + '_set_size-'+ str(args.set_size) + "_setSorting-" + args.setSorting) 
    elif args.removal_metric in ['TauAcc']:
        savepath = os.path.join(savepath, '_tau-'+ str(args.tau),  'metric-' + str(args.removal_metric) + '_normalize-' + str(args.normalize) + '_sortOrder-' + str(args.sortOrder), 'EpochAccMetric-'+ str(args.EpochAccMetric), 
                                'num_sets-'+ str(args.num_sets) + '_set_size-'+ str(args.set_size) + "_setSorting-" + args.setSorting) 
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

    return savepath, loadpath



def load_task_checkpoint(args=None, loadpath=None):
    ckpt = None

    ### If no checkpoint is found, the default value will be None and a new one will be initialized in the Manager
    if args.task_num != 0:
        ### Path to load previous task's checkpoint, if not starting at task 0
        previous_task_path = os.path.join(loadpath, "final.pt") 
        print('path is', previous_task_path)
        ### Reloads checkpoint depending on where you are at for the current task's progress (t->c->p)    
        if os.path.isfile(previous_task_path) == True:
            ckpt = torch.load(previous_task_path)
            print("Checkpoint found and loaded from: ", previous_task_path)
        else:
            print("!!!No checkpoint file found at ", previous_task_path)

    return loadpath, ckpt








def load_pretrained(args, manager):
    if args.task_num==0:
        if args.arch == "vgg16":
            print('\n################################')
            print('\n it is using pretrained model')
            print('\n################################')
            pretrained_state_dict=torch.load('pretrained_model_weights_vgg.pt')
           
            model_state_dict = manager.network.model.state_dict()
            for name, param in pretrained_state_dict.items():
                if 'avgpool' in name:
                    continue  # Skip the average pooling layer
                elif 'classifier1' in name and 'weight' in name:
                    name='features.45.weight'
                elif 'classifier1' in name and 'bias' in name:
                    name='features.45.bias'
                elif 'features' in name:
                    l = name.split('.')
                    name = l[0]+'.'+l[1].split('_')[1]+'.'+ l[2]
                if name in model_state_dict:
                    if model_state_dict[name].shape == param.shape:
                        model_state_dict[name].copy_(param)
                    else:
                        print(f"Shape mismatch for layer {name}, skipping this layer.")
                else:
                    print(f"Layer {name} not found in custom model, skipping this layer.")
        elif args.arch == "modresnet18":
            pretrained_state_dict=torch.load('pretrained_model_weights_mrn18_noaffine.pt')
            model_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'classifier' not in k}
        elif args.arch == "resnet18":
            print("Loading affine resnet18 pretrained from path: ", 'pretrained_model_weights_mrn18_affine.pt', flush=True)
            pretrained_state_dict=torch.load('pretrained_model_weights_mrn18_affine.pt')
            model_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'classifier' not in k}
        elif args.arch == "resnet50":
            pretrained_state_dict=torch.load('pretrained_model_weights_rn50.pt')
            model_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'classifier' not in k}
        
        return model_state_dict










#*# I changed this slightly to handle the assigning of indices 'z' in the main-steps script
def prepare_allbatches(sortMethod=None, set_size=10, dataset=None):

        
    new_indices = torch.arange(len(dataset['y']))

    #*# This is an issue. If we dont also shuffle extraloader then Caper mask won't match the new order of the train dataset shuffled here
    dataset['x'] = dataset['x'][new_indices]
    dataset['y'] = dataset['y'][new_indices]
    dataset['z'] = dataset['z'][new_indices]
    print("Set size in allbatches: ", set_size)
    print("Dataset['x'] size: ", dataset['x'].shape)
    print("Dataset y first 10 samples: ", dataset['y'][:10])
    print("All batches z first 20 samples: ", dataset['z'][:20])

    batches = torch.split(dataset['x'], set_size)
    batch_data = torch.stack(batches)
    # batch_data = torch.stack(torch.split(dataset['x'], set_size))
    batch_labels = torch.stack(torch.split(dataset['y'], set_size))
    batch_IDs = torch.stack(torch.split(dataset['z'], set_size))
    all_batches = list(zip(torch.unbind(batch_data, dim=0), torch.unbind(batch_labels, dim=0), torch.unbind(batch_IDs, dim=0)))   

    total_sum = 0
    labelcount = {}
    for _, labels, _ in all_batches:
        for label in labels:
            if label.item() in labelcount.keys():
                labelcount[label.item()] += 1
            else:
                labelcount[label.item()] = 1
            total_sum += 1

    print("\nOriginal number of samples in sets: ",labelcount)
    print('all_batches sum of lables is', total_sum)

    return all_batches





#####################################################
###    Data Removal Functions
###  Note: We should consolidate these by converting the masks from different functions to a uniform format
#####################################################

def random_remove(all_batches, num_sets, batch_size, cuda=True):
    sets_to_remove = random.sample(range(len(all_batches)), num_sets)
    indices_to_keep = [i for i in range(len(all_batches)) if i not in sets_to_remove]

    x_batches = [all_batches[i][0] for i in indices_to_keep]
    y_batches = [all_batches[i][1] for i in indices_to_keep]
    z_batches = [all_batches[i][2] for i in indices_to_keep]



    # Concatenate along the batch dimension
    x_concatenated = torch.cat(x_batches, dim=0)
    y_concatenated = torch.cat(y_batches, dim=0)
    z_concatenated = torch.cat(z_batches, dim=0)

    # Create a new dataset with the concatenated batches
    train_new_data_loader = list(zip(x_concatenated, y_concatenated, z_concatenated))
    train_new_data_loader = D.DataLoader(train_new_data_loader, batch_size= batch_size, shuffle = True, num_workers = 4, pin_memory=cuda)
    
    return train_new_data_loader, sets_to_remove




def caper_remove(dataset, caper_mask, batch_size, cuda=True):
    batch_data_ca = torch.stack(torch.split(dataset['x'], 1))
    batch_labels_ca = torch.stack(torch.split(dataset['y'], 1))
    batch_IDs_ca = torch.stack(torch.split(dataset['z'], 1))
    ### The static dataset used for calculating HSIC and removing sets. Won't shuffle between calls.
    all_batches_ca = list(zip(torch.unbind(batch_data_ca, dim=0), torch.unbind(batch_labels_ca, dim=0), torch.unbind(batch_IDs_ca, dim=0)))
    x_batches = [all_batches_ca[i][0] for i in caper_mask]
    y_batches = [all_batches_ca[i][1] for i in caper_mask]
    z_batches = [all_batches_ca[i][2] for i in caper_mask]
     # Concatenate along the batch dimension
    x_concatenated = torch.cat(x_batches, dim=0)
    y_concatenated = torch.cat(y_batches, dim=0)
    z_concatenated = torch.cat(z_batches, dim=0)
    # print("Remaining caper IDs: ", z_concatenated)
    # Create a new dataset with the concatenated batches
    train_new_data_loader_caper = list(zip(x_concatenated, y_concatenated, z_concatenated))
    train_new_data_loader_caper = D.DataLoader(train_new_data_loader_caper, batch_size= batch_size, shuffle = True, num_workers = 4, pin_memory=cuda)
    
    return train_new_data_loader_caper

