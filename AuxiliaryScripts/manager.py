"""
Handles all the pruning and connectivity. Pruning steps are adapted from: https://github.com/arunmallya/packnet/blob/master/src/prune.py
Connectivity steps and implementation of connectivity into the pruning steps are part of our contribution
"""
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

import collections
import time
import copy
import random
import multiprocessing
import json
import copy
from math import floor
# from fvcore.nn import FlopCountAnalysis
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler  import MultiStepLR
import torchnet as tnt
import torchattacks
from AuxiliaryScripts.RemovalMetrics.Caper.utils import SmoothCrossEntropyLoss
from AuxiliaryScripts.RemovalMetrics.Caper.checkpoint import save_checkpoint 
# Custom imports
from AuxiliaryScripts import network as net
from AuxiliaryScripts import utils
from AuxiliaryScripts import clmodels
from AuxiliaryScripts.utils import activations
from AuxiliaryScripts.RemovalMetrics import HSIC
from AuxiliaryScripts.Normalization_metrics import Normalization_Techniques
from AuxiliaryScripts import corruptions


import matplotlib.pyplot as plt

class Manager(object):
    """Performs pruning on the given model."""
    ### Relavent arguments are moved to the manager to explicitly show which arguments are used by it
    def __init__(self, args, checkpoint, first_task_classnum=10):
        self.args = args
        self.task_num = args.task_num
        self.train_loader = None 
        self.val_loader = None 
        self.test_loader = None 
        self.hsic_dataset= None
        self.acts_dict = {}
        self.labels = None

        self.save_prefix = None
        
        self.sampledict = None
        ### These are the hardcoded connections withing a ResNet18 model, for simplicity of code. 
        
        ### Note: These are lists indicating subsequent trainable layer pairs
        if args.arch in ["modresnet18", 'resnet18']:
            ### The parent and child idxs for the updated relu layers with in_place=False
            self.parent_idxs = [1, 7, 10, 13, 16, 20, 23, 29, 32, 36, 39, 45, 48, 52, 55, 61]
            self.child_idxs =  [7, 10,13, 16, 20, 23, 29, 32, 36, 39, 45, 48, 52, 55, 61, 64]
            self.act_idxs =    [1,7,10,13,16,20,23,29,32,36,39,45,48,52,55,61,64]

        elif args.arch == "resnet50":
            self.parent_idxs = [1, 7, 9, 11, 18, 20, 22, 26, 28, 30, 35, 37, 39, 46, 48, 50, 54, 56, 58, 62, 64, 66, 71, 73, 75, 82, 84, 86, 90, 92, 94, 98, 100, 102, 106, 108, 110, 114, 116, 118, 123, 125, 127, 134, 136, 138, 142, 144]
            self.child_idxs = [7, 9, 11, 18, 20, 22, 26, 28, 30, 35, 37, 39, 46, 48, 50, 54, 56, 58, 62, 64, 66, 71, 73, 75, 82, 84, 86, 90, 92, 94, 98, 100, 102, 106, 108, 110, 114, 116, 118, 123, 125, 127, 134, 136, 138, 142, 144, 146]
            self.act_idxs = [1, 7, 9, 11, 18, 20, 22, 26, 28, 30, 35, 37, 39, 46, 48, 50, 54, 56, 58, 62, 64, 66, 71, 73, 75, 82, 84, 86, 90, 92, 94, 98, 100, 102, 106, 108, 110, 114, 116, 118, 123, 125, 127, 134, 136, 138, 142, 144, 146]


     
        elif args.arch == "vgg16":
            self.parent_idxs = [2,5,9,12,16,19,22,26,29,32,36,39,42,48]
            self.child_idxs =  [5,9,12,16,19,22,26,29,32,36,39,42,48,51]
            self.act_idxs =    [2,5,9,12,16,19,22,26,29,32,36,39,42,48,51]











        ### Either load from a checkpoint or initialize the necessary masks and network
        if checkpoint != None:
            self.network = checkpoint['network']
            self.all_task_masks = checkpoint['all_task_masks']
        else:
            ### This is for producing and setting the classifier layer for a given task's # classes
            self.network = net.Network(args)
            self.network.add_dataset(str(0), first_task_classnum)
            self.network.set_dataset(str(0))
            
            self.all_task_masks = {}
            task_mask = {}

            for module_idx, (name,module) in enumerate(self.network.model.named_modules()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and name != "classifier":
                    mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
                    mask = mask.cuda()
                    task_mask[module_idx] = mask

            ### Assign a mask for the classifier as well. This is so we can mask away weights pointing to pruned filters
            task_mask["classifier"] = torch.ByteTensor(self.network.model.classifier.weight.data.size()).fill_(1)

            
            self.all_task_masks[0] = task_mask

        
        
        print("\n#######################################################################")
        print("Finished Initializing Manager")
        print("All task Masks keys: ", self.all_task_masks.keys())
        print("Dataset: " + str(self.args.dataset))
        
        print("#######################################################################")

    









     
    """
    ###########################################################################################
    #####
    #####  Connectivity Functions
    #####  Note: You probably dont need any of these
    ###########################################################################################
    """


    ### Run evaluation of val or test loader and get all activations stored as acts_dict
    def calc_activations(self):
        self.network.model.eval()
        self.acts_dict, self.labels, _ = activations(self.val_loader, self.network.model, self.args.cuda, self.act_idxs)
        self.network.model.train()
        print("Done collecting activations")


    ### Calculate connectivities for the full network without masking
    def calc_conns(self):
        all_conns = {}
        self.calc_activations()
    
        ### The keys for all_conns are labeled as a range because both parent and child idxs have duplicate entries. These keys match to indices in those dictionaries, not module idxs 
        for key_id in range(0,len(self.parent_idxs)): 
            all_conns[key_id] = self.calc_conn(self.parent_idxs[key_id], self.child_idxs[key_id], key_id)
        return all_conns


    def calc_conn(self, parent_key, child_key, key_id):
        p1_op = {}
        c1_op = {}

        p1_op = copy.deepcopy(self.acts_dict[parent_key]) 
        c1_op = copy.deepcopy(self.acts_dict[child_key])

        parent_aves = []
        p1_op = p1_op.numpy()
        c1_op = c1_op.numpy()
        
        if np.count_nonzero(np.isnan(p1_op)) > 0 or np.count_nonzero(np.isnan(c1_op)) > 0:
            print("Raw activations are nan")
            
        ### Connectivity is standardized by class mean and stdev
        for label in list(np.unique(self.labels.numpy())):
            parent_mask = np.ones(p1_op.shape,dtype=bool)
            child_mask = np.ones(c1_op.shape,dtype=bool)

            parent_mask[self.labels != label] = False
            parent_mask[:,np.all(np.abs(p1_op) < 0.0001, axis=0)] = False
            child_mask[self.labels != label] = False
            child_mask[:,np.all(np.abs(c1_op) < 0.0001, axis=0)] = False
            
            p1_op[parent_mask] -= np.mean(p1_op[parent_mask])
            p1_op[parent_mask] /= np.std(p1_op[parent_mask])

            c1_op[child_mask] -= np.mean(c1_op[child_mask])
            c1_op[child_mask] /= np.std(c1_op[child_mask])



        """
        Code for averaging conns by parent prior by layer
        """
        parent_class_aves = []
        parents_by_class = []
        parents_aves = []
        conn_aves = []
        parents = []
        for cl in list(np.unique(self.labels.numpy())):
            p1_class = p1_op[self.labels == cl]
            c1_class = c1_op[self.labels == cl]

            ### Parents is a 2D list of all of the connectivities of parents and children for a single class
            coefs = np.corrcoef(p1_class, c1_class, rowvar=False).astype(np.float32)
            parents = []
            ### Loop over the cross correlation matrix for the rows corresponding to the parent layer's filters
            for i in range(0, len(p1_class[0])):
                ### Append the correlations to all children layer filters for the parent filter i. We're indexing the upper-right quadrant of the correlation matrix between x and y
                #!# Nans: If a parent is omitted, this entire set will be NaN, if a child is omitted, then only the corresponding correlation is nan
                ###    Note: These NaNs are expected and not an issue since they dont appear in the indexed values for the current subnetwork/task
                parents.append(coefs[i, len(p1_class[0]):])
            ### We take the absolute value because we only care about the STRENGTH of the correlation, not the 
            parents = np.abs(np.asarray(parents))

            ### This is a growing list of each p-c connectivity for all activations of a given class
            ###     The dimensions are (class, parent, child)
            parents_by_class.append(parents)
        
        conn_aves = np.mean(np.asarray(parents_by_class), axis=0)
        
        return conn_aves
        





    def prepare_task(self):
        self.make_taskmask()
        ### Have to add the new classifier's task mask AFTER the classifier has been added to the network at the start of the task
        ### Reload the previously omitted frozen weights. We don't want to reload the previous tasks newly trained weights so we offset by 1 task to only target omitted weights
        self.update_statedict(tasknum=(self.task_num - 1))
        ### Update the backup model to reflect the newly trained weights and batchnorms before making sharing decision
        self.network.backupmodel.load_state_dict(self.network.model.state_dict())
        ### Decide which frozen weights to mask to zero for the task
        self.pick_shared_task()
        self.reinit_statedict()









    ### This function was written to determine which past tasks to share weights from. I've gutted it to simplify it and share all past weights for each task since its not relevant to the project
    def pick_shared_task(self):
        print("Current tasknum: ", self.task_num)
        for module_idx, (name,module) in enumerate(self.network.model.named_modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and name != "classifier":

                new_weights = utils.get_trainable_mask(module_idx, self.all_task_masks, self.task_num)
                
                ### Alternatively just identify frozen filters and prevent new weights being connected to them. Much more robust and simple
                frozen_filters = utils.get_frozen_mask(module.weight.data, module_idx, self.all_task_masks, self.task_num)

                # print("Number of frozen filters: ", frozen_filters.long().sum())
                new_weights[frozen_filters.eq(1)] = 0
                
                ### This will omit any weights which weren't used in the task being shared while keeping all trainable weights
                self.all_task_masks[self.task_num][module_idx] = new_weights
                
                for t in range(0,self.task_num):
                    shared_weights = self.all_task_masks[t][module_idx].clone().detach()
                    ### This will omit any weights which weren't used in the task being shared while keeping all trainable weights
                    self.all_task_masks[self.task_num][module_idx] = torch.max(new_weights, shared_weights)
            
        
        ### Set all omitted weights to 0 with the updated task mask
        self.network.apply_mask(self.all_task_masks, self.task_num)

        
    """
    ##########################################################################################################################################
    Pruning Functions
    ##########################################################################################################################################
    """
    ### Goes through and calls prune_mask for each layer and stores the results
    ### Then applies the masks to the weights
    def prune(self):
        print('Pruning for dataset idx: %d' % (self.task_num))
        print('Pruning each layer by removing %.2f%% of values' % (100 * self.args.prune_perc_per_layer))
        
        for module_idx, (name,module) in enumerate(self.network.model.named_modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and name != "classifier":

                trainable_mask = utils.get_trainable_mask(module_idx, self.all_task_masks, self.task_num)
              
                ### Get the pruned mask for the current layer
                pruned_mask = self.pruning_mask(module.weight.data.clone().detach(), trainable_mask, module_idx)

                for module_idx2, (name2,module2)  in enumerate(self.network.backupmodel.named_modules()):
                    if module_idx == module_idx2:
                        print("Module idx 2:", module_idx2, " ", module2, flush=True)
                        module2.weight.data[pruned_mask.eq(1)] = 0.0
                        
                # Set pruned weights to 0.
                module.weight.data[pruned_mask.eq(1)] = 0.0
                self.all_task_masks[self.task_num][module_idx][pruned_mask.eq(1)] = 0


            

 
    def pruning_mask(self, weights, trainable_mask, layer_idx):
        
        weight_magnitudes = weights.abs()
        trainable_mask = trainable_mask.eq(1)
        task_mask = self.all_task_masks[self.task_num][layer_idx].eq(1)


        ### Calculate the number of incoming weights that haven't been omitted for each filter prior to averaging
        weights_num = trainable_mask.long().sum()
        ### This is the average weight values for ALL filters in current layer (not counting omitted incoming weights)

        included_weights = weight_magnitudes[trainable_mask]

        prune_sparsity = self.args.prune_perc_per_layer
            
        
        ### Now we use our masked set of averaged 1D feature weights to get a pruning threshold
        prune_rank = round(prune_sparsity * included_weights.numel())

        prune_value = included_weights.view(-1).cpu().kthvalue(prune_rank)[0]

        ### Now that we have the pruning threshold, we need to get a mask of all filters who's average incoming weights fall below it        
        weights_to_prune = weight_magnitudes.le(prune_value)

            
        prune_mask = torch.zeros(weights.shape)
        prune_mask[weights_to_prune]=1        
                        
        ### Prevent pruning of any non-trainable weights (frozen or omitted)
        prune_mask[trainable_mask.eq(0)]=0
    
            
        ### Check how many weights are being chosen for pruning
        print('Layer #',layer_idx, ' pruned ',prune_mask.eq(1).sum(), '/', prune_mask.numel() ,
                '(',100 * prune_mask.eq(1).sum() / prune_mask.numel(),'%%)', ' (Total in layer: ', weights.numel() ,')')

        return prune_mask
        
        







    def eval(self, num_class=10, use_attack=True, Data=None):
        """Performs evaluation with per-class accuracy tracking for non-attacked predictions."""

        self.network.apply_mask(self.all_task_masks, self.task_num)
        self.network.model.eval()

        error_meter = None
        error_meter_attacked = None

        for batchidx, (batch, label, IDs) in enumerate(Data):
            if self.args.cuda:
                batch = batch.cuda()
                label = label.cuda()





            if use_attack:
                if self.args.attack_type == 'PGD':
                    attack = torchattacks.PGD(self.network.model, eps=0.03, alpha=0.0039, steps=10, random_start=True)
                    x_attacked = attack(batch, label)
                elif self.args.attack_type == 'AutoAttack':
                    print("Using autoattack in eval")
                    attack = torchattacks.AutoAttack(self.network.model, norm='Linf', eps=8/255, version='standard', n_classes=num_class, seed=None, verbose=False)
                    x_attacked = attack(batch, label)
                elif self.args.attack_type in ["gaussian_noise", "gaussian_blur", "saturate", "rotate"]:
                    x_attacked = batch.clone()
                    for i in range(len(batch)):
                        x_attacked[i] = getattr(corruptions, self.args.attack_type)(batch[i], severity=3)
                else:
                    x_attacked = batch

                if self.args.cuda:
                    x_attacked = x_attacked.cuda()

            # Forward pass on non-attacked batch
            output = self.network.model(batch)
            preds = output.argmax(dim=1)  # Get predicted class indices

            # Track overall accuracy
            if error_meter is None:
                topk = [1, 5] if output.size(1) > 5 else [1]
                error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter.add(output.data, label)

            # Forward pass on attacked batch if applicable
            if use_attack:
                output_attacked = self.network.model(x_attacked)
                if error_meter_attacked is None:
                    topk_attacked = [1, 5] if output_attacked.size(1) > 5 else [1]
                    error_meter_attacked = tnt.meter.ClassErrorMeter(topk=topk_attacked)
                error_meter_attacked.add(output_attacked.data, label)

        self.network.model.train()

        # Compute overall error
        errors = error_meter.value()
        

        if use_attack:
            errors_attacked = error_meter_attacked.value()
            return errors, errors_attacked
        else:
            return errors








    ### Train the model for the current task, using all past frozen weights as well
    def train(self, epochs=10, save=True, savename='', num_class=10, total_epochs = 0, start_epoch = 0, use_attack=True, save_best_model=False, trackpreds = False):
        if total_epochs == 0:
            total_epochs = self.args.train_epochs
        """Performs training."""
        print('\n\n number of epochs is', epochs)
        best_model = None
        best_model_acc = 0
        best_model_adv_acc = 0
        val_acc_history = []
        val_acc_history_attacked = []

        if self.args.cuda:
            self.network.model = self.network.model.cuda()

        # Get optimizer with correct params.
        params_to_optimize = self.network.model.parameters()

        lr = self.args.lr
        optimizer = optim.SGD(params_to_optimize, lr=lr, momentum=0.9, weight_decay=0.0000, nesterov=True)
        
        
        if self.args.use_train_scheduler:
            # milestones = [40,60,80]
            ### These weren't tuned, just rough estimates aimed to allow convergence at each step for either training or finetuning steps
            # for 300: [120, 180, 240], for 150: [60, 90, 120]
            milestones = [(2*(epochs/5)),(3*(epochs/5)),(4*(epochs/5))]

            scheduler  = MultiStepLR(optimizer, milestones=milestones, gamma=self.args.Gamma) 
        else:
            patience = self.args.lr_patience
            lrmin = self.args.lr_min

        
        loss = nn.CrossEntropyLoss()
        # loss = SmoothCrossEntropyLoss(smoothing=self.args.caper_epsilon)

        self.network.model.train()
        start_time = time.time()

      
        hsic_list_layerwise = {}
        cka_list_layerwise = {}
        if start_epoch==0:
            epochs_to_run = range(epochs)
        else: 
            epochs_to_run= range(start_epoch-1, epochs+start_epoch-1)

         # Initialize total FLOPs counter
        # total_flops = 0
        for idx in epochs_to_run:
            epoch_idx = idx + 1
            # print('Epoch: ', epoch_idx, ' Learning rate:', optimizer.param_groups[0]['lr'], flush=True)
            total_correct = 0  # Track total correct predictions
            total_samples = 0  # Track total number of samples
            
            for batch_id, (x, y, z) in enumerate(self.train_loader):
        
                if self.args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    # z = z.cuda()
                x = Variable(x)
                y = Variable(y)



                if use_attack==True:

                    if self.args.attack_type in ['PGD', 'AutoAttack']:
                        attack = torchattacks.PGD(self.network.model, eps=0.03, alpha=0.0039, steps=10, random_start=True)
                        x = attack(x,y)

                    elif self.args.attack_type == "gaussian_noise":
                        for i in range(len(x)):
                            x[i] = corruptions.gaussian_noise(x[i], severity=3)

                    elif self.args.attack_type == "impulse_noise":
                        for i in range(len(x)):
                            x[i] = corruptions.impulse_noise(x[i], severity=3)

                    elif self.args.attack_type == "gaussian_blur":
                        for i in range(len(x)):
                            x[i] = corruptions.gaussian_blur(x[i], severity=3)

                    elif self.args.attack_type == "spatter":
                        for i in range(len(x)):
                            x[i] = corruptions.spatter(x[i], severity=3)

                    elif self.args.attack_type == "saturate":
                        for i in range(len(x)):
                            x[i] = corruptions.saturate(x[i], severity=3)

                    elif self.args.attack_type == "rotate":
                        for i in range(len(x)):
                            x[i] = corruptions.rotate(x[i], severity=3)


                if self.args.cuda:
                    x = x.cuda()
              
    
                # Set grads to 0.
                self.network.model.zero_grad()
        
                # Do forward-backward.
                output = self.network.model(x)



                ### If tracking the predictions made, then we store the logits for each sample on the current epoch
                if trackpreds:
                    for n, sampleID in enumerate(z):
                        self.sampledict['epochlogits'][sampleID.item()].append(output[n]) 
                        self.sampledict['labels'][sampleID.item()] = y[n] 


                loss(output, y).backward()

                # Set frozen param grads to 0.
                self.network.make_grads_zero(self.all_task_masks, self.task_num)
                
                # Update params.
                optimizer.step()

                # Track training accuracy
                preds = output.argmax(dim=1)  # Get predicted class indices
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)  # Count number of samples in batch

            

            if self.args.use_train_scheduler:
                scheduler.step()
            end_time = time.time()
            

            ### Every Nth epoch evaluate. Compromises between seeing periodic accuracy updates and the added runtime of making attacks on test data
            ### If you want accuracies to be printed out each epoch use verbose==True
            if epoch_idx % self.args.eval_interval == 0 or epoch_idx==epochs:
                # Compute and print training accuracy
                train_accuracy = 100.0 * total_correct / total_samples
                print(f"Epoch {epoch_idx} - Training Accuracy: {train_accuracy:.2f}%")
            

                print('\nEvaluating at Epoch: ', epoch_idx, ' after ', self.args.eval_interval, 
                        ' epochs at Learning rate:', optimizer.param_groups[0]['lr'], flush=True)
                if epoch_idx == epochs:
                    print("Final epoch evaluation: epoch ", epoch_idx)

                val_errors, val_errors_attacked = self.eval(num_class,  use_attack=True, Data=self.val_loader)
                val_acc_history.append(100-val_errors[0])
                val_acc_history_attacked.append(100-val_errors_attacked[0])
                val_accuracy = 100 - val_errors[0]  # Top-1 accuracy.
                val_accuracy_attacked = 100 - val_errors_attacked[0]  # Top-1 accuracy.
              

               
                print('Adversarial Accuracy on attacked data: %0.2f%%, best is %0.2f%%' %(val_accuracy_attacked, best_model_adv_acc))  
                print('Vanilla Accuracy on un-attacked data: %0.2f%%, best is %0.2f%%' %(val_accuracy, best_model_acc))

                if self.args.use_train_scheduler == False:
                    if ((use_attack == True and val_accuracy_attacked > best_model_adv_acc)
                        or (use_attack == False and val_accuracy > best_model_acc)):
                        best_model_adv_acc = val_accuracy_attacked 
                        # best_model_acc might not be the best, as we need the best model based on adversarial accuracy not the normal one
                        best_model_acc = val_accuracy
                        ### If using early stopping instead of scheduling, reset patience when the accuracy is improved and save the new best model weights
                        best_model = copy.deepcopy(self.network.model.state_dict())
                        patience = self.args.lr_patience

                    else:
                        patience -= self.args.eval_interval
                        ### After sufficient epochs with no accuracy improvement, lr decays. If its below the min threshold, then the model is expected to have converged and training ends early
                        if patience <= 0:
                            lr *= self.args.lr_factor
                            if lr < lrmin:
                                break
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= self.args.lr_factor  
                            patience = self.args.lr_patience                        
                else:
                    if ((use_attack == True and val_accuracy_attacked > best_model_adv_acc)
                        or (use_attack == False and val_accuracy > best_model_acc)):
                        best_model_adv_acc = val_accuracy_attacked  
                        best_model_acc = val_accuracy
                        best_model = copy.deepcopy(self.network.model.state_dict())
                                        
                    
        print('Finished finetuning...')
        print('normal acc of the best model: %0.2f%%, attacked acc of the best model:%0.2f%%' %(best_model_acc, best_model_adv_acc))
        print('-' * 16)

        if best_model != None:
            self.network.model.load_state_dict(copy.deepcopy(best_model))
        
        return best_model_acc, best_model_adv_acc, val_acc_history, val_acc_history_attacked








        
    """
    ##########################################################################################################################################
    Functions for Weight and Masking Modifications
    ##########################################################################################################################################
    """


    def reinit_statedict(self):
        print("\nReinitializing trainable zeroed weights")
        ### Reset the weights of the initialization model
        if self.args.arch == "modresnet18":
            self.network.initmodel = clmodels.modifiedresnet18()
        elif self.args.arch == "resnet18":
            self.network.initmodel = clmodels.modifiedresnet18(affine=True)
        elif self.args.arch == "resnet50":
            self.network.initmodel = clmodels.modifiedresnet50(affine=True)

        for module_idx, (name,module) in enumerate(self.network.model.named_modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and name != "classifier":

                new_weights = utils.get_trainable_mask(module_idx, self.all_task_masks, self.task_num)
                ### Get the frozen filters and use to mask out any previously omitted weights which appear as trainable                
                frozen_filters = utils.get_frozen_mask(module.weight.data, module_idx, self.all_task_masks, self.task_num)

                new_weights[frozen_filters.eq(1)] = 0

                for module_idx2, (name2,module2) in enumerate(self.network.initialmodel.named_modules()):
                    if module_idx2 == module_idx:
                        module.weight.data[new_weights.eq(1)] = module2.weight.data.clone()[new_weights.eq(1)]


        

    ### Reloads the values of all frozen weights in order to undo any omitting. This is done usually when moving to a new task to allow for new masks to be selected
    def update_statedict(self, tasknum, use_trainable=False):
        print("Updating state dict from task number: ", tasknum)
        for module_idx, (name,module) in enumerate(self.network.model.named_modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and name != "classifier":

                if use_trainable == False:
                    frozenmask = utils.get_frozen_mask(module.weight.data, module_idx, self.all_task_masks, tasknum)
                else:
                    trainable_mask = utils.get_trainable_mask(module_idx, self.all_task_masks, tasknum)
                    frozenmask = torch.zeros(trainable_mask.shape)
                    ### Set the frozen mask as the compliment of the trainable mask, this way weights from after the given task tasknum will be reloaded. 
                    ###       This is to be used after revisiting a past task for training and updating the remaining weights afterwards.
                    frozenmask[trainable_mask.eq(0)] = 1
                            
                for module_idx2, (name2,module2) in enumerate(self.network.backupmodel.named_modules()):
                    if module_idx2 == module_idx:
                        module.weight.data[frozenmask.eq(1)] = module2.weight.data.clone()[frozenmask.eq(1)]
    
       
       

    ### Makes the taskmask for a newly encountered task
    def make_taskmask(self):
        ### Creates the task-specific mask during the initial weight allocation
        task_mask = {}
        for module_idx, (name,module) in enumerate(self.network.model.named_modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and name != "classifier":

                task = torch.ByteTensor(module.weight.data.size()).fill_(1)
                task = task.cuda()
                task_mask[module_idx] = task

        ### Initialize the new tasks' inclusion map with all 1's
        self.all_task_masks[self.task_num] = task_mask
        
        self.all_task_masks[self.task_num]["classifier"] = torch.ByteTensor(self.network.model.classifier.weight.data.size()).fill_(1)
        
        print("Exiting finetuning mask")

       
       
       
