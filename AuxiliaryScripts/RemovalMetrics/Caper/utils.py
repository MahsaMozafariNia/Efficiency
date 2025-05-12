import torch
import copy
import pickle
import random 

import numpy    as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import norm
from pathlib import Path
from torch.nn.modules.loss import _WeightedLoss
import datetime

#import matplotlib.mlab as mlab
#import matplotlib as mpl
#mpl.use('Agg')

visualisation = {}

"""
Code Acknowledgement: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
Code Acknowledgement: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
"""

#### Label Smoothing Loss (Pytorch Native)####
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight    = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),device=targets.device) \
                                  .fill_(smoothing / (n_classes-1)).scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


#### Linear Scaling Function ####
def terp(min_value, max_value, value):
    return ((value - min_value)/(max_value - min_value))

#### Hook Function
def hook_fn(m, i, o):
    visualisation[m] = o 


#### Return Forward Hooks To All Layers
def get_all_layers(net, hook_handles, item_key):
 
    # for name, layer in net._modules.items():
        ################
        ### to be compatible for our model (in our model we have two sequentials features and classifiers but in Caper model, the structure is different)
        ################
    for name, layer in net.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if(name+'.weight' == item_key):
                hook_handles.append(layer.register_forward_hook(hook_fn))
        # END IF

    # END FOR

#### Generate and collect activations from specific layer ####
def generate_activations(data_loader, model, device, item_key, save_act=False, save_act_fname='norm', save_id_fname='idnorm', ret_act=True, mid_level=False, f_idx=None, sampledict=None, perturbed=False, trial=-1):
    temp_op            = [] #None
    parents_op       = [] #None
    handles     = []

    get_all_layers(model, handles, item_key)
  

    if save_act:
        save_handle = open(save_act_fname, 'ab+') # Open/Create file in appending format
        # save_id_handle = open(save_id_fname, 'ab+') # Open/Create file in appending format
        
        ### If we aren't currently generating the original values, it will load them as well
        if Path(save_act_fname.replace('.txt','_orig.txt')).is_file():
            orig_handle = open(save_act_fname.replace('.txt','_orig.txt'), 'rb')


    # print('Collecting Activations for Layer %s'%(item_key))

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label, z_label = data
            ops = model(x_input.to(device), labels=True)
           
            #!# Changed to properly record normal logits before the trials
            if mid_level == False and trial in [-1,0]:
                if perturbed:
                    for i, ID in enumerate(z_label):
                        sampledict['tau_advlogits'][ID.item()].append(ops[i]) 
                else:
                    # print("Caper adding tau logits")
                    for i, ID in enumerate(z_label):
                        sampledict['tau_logits'][ID.item()].append(ops[i]) 




            temp_op = visualisation[list(visualisation.keys())[0]].cpu().numpy()


            ### Given a list of most-sensitive filters, only keep the activations from those filters
            if not(f_idx is None):
                # print("F_idx is: ", f_idx, " temp_op shape: ", temp_op.shape, flush=True)
                temp_op = temp_op[:, f_idx]


            if Path(save_act_fname.replace('.txt','_orig.txt')).is_file():
                act_orig = pickle.load(orig_handle)
                # print("Act orig shape: ", act_orig.shape, flush=True)
                temp_op = act_orig - temp_op


            pickle.dump(temp_op, save_handle)
            # pickle.dump(id_op, save_id_handle)


    save_handle.close()
    
    if Path(save_act_fname.replace('.txt','_orig.txt')).is_file():
        orig_handle.close()


    for handle in handles:
        handle.remove()    
    
    del visualisation[list(visualisation.keys())[0]]


    if ret_act:
        return temp_op, None, None, sampledict#labels_op, pred_labels_op

    else:
        return None, None, None, sampledict













#### Function uses the determined key heuristic to define Error Prone Filters ####
def get_epf_idx(act_norm, act_noise):
    # Assumption: act_norm = Samples X F X H X W, act_noise = Trials X Samples X F X H X W
    MIN = np.min(np.linalg.norm(np.linalg.norm(np.linalg.norm(act_norm - act_noise, axis=3),axis=3),axis=1))
    MAX = np.max(np.linalg.norm(np.linalg.norm(np.linalg.norm(act_norm - act_noise, axis=3),axis=3),axis=1))

    # Since min and max were computed over l2-norm of H, W and Samples, those values must be linearly scaled
    mid_data =  np.mean(terp(MIN, MAX, np.linalg.norm(np.linalg.norm(np.linalg.norm(act_norm - act_noise, axis=3),axis=3),axis=1)), 0)

    assert(np.min(mid_data)==0 and np.max(mid_data)==1)
    
    # 10% as threshold to define EPFs
    threshold_epf = (np.max(mid_data) - np.min(mid_data))*0.1 + np.min(mid_data)
    epf_idx       = np.where(mid_data > threshold_epf)[0]

    return epf_idx


#### Function to use EPFs to count the samples which fall beyond desired threshold ####
def get_thresholded_samples(act_norm, act_noise, save_plot=False, save_plot_fname='layer1'):

    # Assumption: act_norm = Samples X F X H X W, act_noise = Trials X Samples X F X H X W
    sample_counter = {}
    for idx in range(act_norm.shape[0]):
        sample_counter[str(idx)] = 0

    # END FOR

    epf_idx = get_eps_idx(act_norm, act_noise)

    MIN = np.min(np.linalg.norm(np.linalg.norm(act_norm - act_noise, axis=3),axis=3))
    MAX = np.max(np.linalg.norm(np.linalg.norm(act_norm - act_noise, axis=3),axis=3))

    # Loop over each EPF and count samples beyond thresholds
    for each_epf_idx in epf_idx:
        # Since min and max were computed over l2-norm of H and W, those values must be linearly scaled
        mid_data =  np.mean(terp(MIN, MAX, np.linalg.norm(np.linalg.norm(act_norm[:,each_epf_idx,:,:] - act_noise[:,:,each_epf_idx,:,:], axis=2),axis=2)), 0)

        assert(np.min(mid_data)==0 and np.max(mid_data)==1)

        # Fit a normal distribution to the data
        mu, std = norm.fit(mid_data)

        # Generate histogram of heuristic
        counts, bin_edges, _ = plt.hist(mid_data, bins=np.linspace(0,1,1000), alpha=0.6, density=True, color='b')
        # Generate normalized counts from histogram
        norm_counts  = counts*np.diff(bin_edges)[0]*data.shape[0]
        # Generate bin values from bin edges 
        norm_bin_idx = np.diff(bin_edges)/2. + bin_edges[:999]

        # Generate thresholds to count samples
        d_p_threshold  = mu + std
        d_m_threshold  = mu - std
        
        # Count samples beyond thresholds
        high_energy_counts = np.sum(norm_counts[norm_bin_idx >= d_p_threshold]) 
        low_energy_counts  = np.sum(norm_counts[norm_bin_idx <= d_m_threshold]) 
        all_counts         = np.concatenate((low_energy_counts, high_energy_counts))

        for counts in all_counts:
            sample_counter[str(counts)] += 1

        # END FOR
        
        if save_plot:
            prob = norm.pdf(norm_bin_idx, mu, std)
            RMSE = np.sqrt(np.mean(np.square(counts - prob)))
            y    = mlab.normpdf(bin_edges, mu, std)

            plt.plot(bin_edges, y, 'k--', linewidth=2)
            title = "Fit results: mu = %.2f,  std = %.2f, RMSE= %.2f" % (mu, std, RMSE)
            plt.title(title)
            plt.vlines(mu+std, 0, max(counts))
            plt.vlines(mu-std, 0, max(counts))
            plt.savefig('INS_data/images/'+save_plot_fname+'_'+str(each_epf_idx)+'.png')
        # END IF

        plt.clf()

    # END FOR    
    return sample_counter
