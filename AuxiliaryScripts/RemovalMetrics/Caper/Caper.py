import copy
import torch
import scipy
import pickle
import random
import argparse
import time
import numpy             as np
import torch.nn          as nn
import torch.optim       as optim
from torch.optim.lr_scheduler  import MultiStepLR, CosineAnnealingLR
from AuxiliaryScripts.RemovalMetrics.Caper.checkpoint import save_checkpoint 
from AuxiliaryScripts.RemovalMetrics.Caper.utils import generate_activations, SmoothCrossEntropyLoss 
from AuxiliaryScripts.Normalization_metrics import Normalization_Techniques
import os
from AuxiliaryScripts import corruptions 
import datetime

class Caper_Method():
    
    def __init__(self, args, model, extra_loader, sampledict):
        self.args = args
        self.model = model
        self.extraloader= extra_loader
        self.sampledict = sampledict


    ##### Generate New Data_Loader
    def generate_data_batches(self, data, labels, IDs, batch):
        new_list_data   = []
        new_list_labels = []
        new_list_IDs = []
        # print("In caper generate data batches, data shape: ", data.shape[0])

        for gen_batch_idx in np.arange(0,data.shape[0],batch):
            if gen_batch_idx + batch < data.shape[0]:
                new_list_data.append(torch.Tensor(data[gen_batch_idx:gen_batch_idx+batch]))
                new_list_labels.append(torch.Tensor(labels[gen_batch_idx:gen_batch_idx+batch]))
                new_list_IDs.append(torch.Tensor(IDs[gen_batch_idx:gen_batch_idx+batch]))
                # print("newlist IDs: ", new_list_IDs)
            else:
                new_list_data.append(torch.Tensor(data[gen_batch_idx:]))
                new_list_labels.append(torch.Tensor(labels[gen_batch_idx:]))
                new_list_IDs.append(torch.Tensor(IDs[gen_batch_idx:]))

        # print("New List IDs: ", new_list_IDs)
        new_data_loader = zip(new_list_data, new_list_labels, new_list_IDs)

        return new_data_loader



    def generate_stats(self, perturb_handle, id_handle, size=391):
        print("Size: ", size)
        cur_var = []
        # id_var = []

        #for trial in range(trials):
        perturb_f = open(perturb_handle, 'rb')
        # id_f = open(id_handle, 'rb')

        firstflag = False
        try:
            while True:
                perturb = pickle.load(perturb_f) 
                cur_var.append(np.linalg.norm(perturb, axis=(2,3)))

                # if firstflag == False:
                #     print("Perturb shape: ", len(perturb), " ", perturb)
                #     firstflag = True


                # ### Load corresponding IDs
                # ids = pickle.load(id_f)  # Shape: [128, 16, 2, 2]
                
                # ### Step 2: Check if all IDs match in the 2x2 spatial dimensions
                # for batch_idx in range(ids.shape[0]):  # Iterate over each sample in the batch
                #     unique_ids = np.unique(ids[batch_idx])  # Get unique IDs for this sample
                #     if len(unique_ids) != 1:
                #         print(f"Warning: Mismatched IDs in batch index {batch_idx}, unique IDs: {unique_ids}")

                # ### Step 3: Collapse redundant dimensions (2x2) to a single integer per filter
                # ids_collapsed = ids[:, :, 0, 0]  # Shape: [128, 16]

                # ### Append to the ID list
                # id_var.append(ids_collapsed)  # Each element in id_var is [128, 16]

        except EOFError:
            pass

        perturb_f.close()
        # id_f.close()


        cur_var = np.concatenate(cur_var)
        print("In generate stats current var shape: ", cur_var.shape)
        if len(cur_var.shape) < 3:
            cur_var = np.expand_dims(cur_var, 0)

        # id_var = np.concatenate(id_var)
        # print("In generate stats current id shape: ", id_var.shape)
        # if len(id_var.shape) < 3:
        #     id_var = np.expand_dims(id_var, 0)

        cur_reshaped = np.reshape(cur_var, (int(cur_var.shape[1]/size), size, -1))
        print("In generate stats current reshaped var shape: ", cur_reshaped.shape)

        # id_reshaped = np.reshape(id_var, (int(id_var.shape[1]/size), size, -1))
        # print("In generate stats current reshaped id shape: ", id_reshaped.shape)


        # print("First 20 IDs of each trial:", id_reshaped[:,:20,0], flush=True)

        # # Step 5: Verify ID consistency across trials
        # for sample_idx in range(id_reshaped.shape[1]):  # Iterate over samples (dim=1)
        #     # Get the unique IDs across all trials and filters for this sample
        #     unique_ids = np.unique(id_reshaped[:, sample_idx, :])  # Shape: [trials, filters]
        #     if len(unique_ids) != 1:
        #         print(f"Warning: Mismatched IDs for sample {sample_idx}, unique IDs: {unique_ids}")





        cur_var = np.mean(cur_reshaped,0)
        print("In generate stats current var shape: ", cur_var.shape)
        torch.save(torch.from_numpy(cur_var), (self.args.save_prefix + "/caper_mean_act_difs.pt"))

        # id_var = np.mean(id_reshaped,0)
        # print("In generate stats current id shape: ", id_var.shape)
        # print("In generate stats current id first 10 IDs: ", id_var[:10])

        cur_min = np.repeat(np.min(cur_var,0), cur_var.shape[0]).reshape(cur_var.shape[1], cur_var.shape[0]).T
        cur_max = np.repeat(np.max(cur_var,0), cur_var.shape[0]).reshape(cur_var.shape[1], cur_var.shape[0]).T
        Normalization = Normalization_Techniques(self.args)
        norm_cur_var = Normalization.interp(cur_min, cur_max, cur_var)
        ### Seems like it averages over all trials for each given sample, so the final shape should be the number of samples by the number of tracked filters
        mid_data     = np.mean(norm_cur_var, 1)
        print("mid data shape: ", mid_data.shape)

        return mid_data




    #### Collect Statistics from input perturbation differences to generate curriculum ####
    def generate_counts(self, model, extraloader, device, layer, child_layer, batch, trials=10, save_act_fname='TEMP_FNAME', save_id_fname='TEMP_FNAME'):
        # Set model to eval mode
        model.eval()

        print("\n\n\nStarting generate_counts: ", datetime.datetime.now(), flush=True)


        ### If rerunning an existing experiment, just makes sure we aren't appending new values to the existing output files
        if os.path.exists(save_act_fname):
            os.remove(save_act_fname)  # Remove the main activation file
            print(f"Cleared file: {save_act_fname}")
        if os.path.exists(save_act_fname.replace('.txt', '_orig.txt')):
            os.remove(save_act_fname.replace('.txt', '_orig.txt'))  # Remove the '_orig' activation file
            print(f"Cleared file: {save_act_fname.replace('.txt', '_orig.txt')}")

        # if os.path.exists(save_id_fname):
        #     os.remove(save_id_fname)  # Remove the main activation file
        #     print(f"Cleared file: {save_id_fname}")

        act_orig    = None
        act_perturb = None
        size        = None

        print('Starting Time Trial')
        s_time = time.time()

        # Sensitivty prior to curate a subset of important filters used to gather activations
        p_layer = np.mean(model.state_dict()[layer].detach().cpu().numpy(), (2,3))
        
        #!# Added for resnet18 to handle fully connected child layers. Alternative can use the penultimate conv layer instead
        c_layer = model.state_dict()[child_layer].detach().cpu().numpy()
        if len(c_layer.shape)>2:
            c_layer = np.mean(c_layer, (2,3))
        else:
            print("not averaging the child activations for child layer: ", child_layer)
        #if p_layer.shape[0] != c_layer.shape[1]: 
        #    if p_layer.shape[0] < c_layer.shape[1]:
        #        c_layer = c_layer[:, -p_layer.shape[0]:]

        ### Normalize all filters per-sample, then average over all samples per-filter
        norm_const = np.sum(c_layer, 1)
        sens_prior = c_layer/np.repeat(norm_const, c_layer.shape[1]).reshape(c_layer.shape)
        sens_prior = np.mean(sens_prior, 0)
        
        Normalization = Normalization_Techniques(self.args)
        sens_prior = Normalization.interp(np.min(sens_prior), np.max(sens_prior), sens_prior)

        assert(np.max(sens_prior) <= 1.)
        assert(np.min(sens_prior) >= 0.)

        ### Get the 16 most sensitive filters to track when determining susceptibility of samples to perturbation
        sens_idx = np.argsort(sens_prior)[-16:]
        # sens_idx = np.argsort(sens_prior)[-64:]

        print("Starting generate_activations original: ", datetime.datetime.now(), flush=True)

        # Generate activations for original data
        _, _, _, self.sampledict = generate_activations(copy.deepcopy(extraloader), self.model, device, layer, save_act=True, save_act_fname=save_act_fname.replace('.txt', '_orig.txt'), save_id_fname=save_id_fname.replace('.txt', '_orig.txt'), ret_act=False, f_idx=sens_idx, sampledict=self.sampledict)

        for trial in range(trials):
            print("Starting trial: ", datetime.datetime.now(), flush=True)
            print("Starting counts for trial: ", trial, flush=True)
            #### Generate perturbed data ####
            perturb_data = [x for (x, y, z) in extraloader]
            orig_labels  = [y for (x, y, z) in extraloader]
            orig_IDs     = [z for (x, y, z) in extraloader]
            perturb_data = torch.cat(perturb_data,0)
            orig_labels  = torch.cat(orig_labels,0).type(torch.FloatTensor)
            orig_IDs     = torch.cat(orig_IDs,0)




            for perturb_data_idx in range(perturb_data.shape[0]):
                temp_input   = perturb_data[perturb_data_idx]
                temp_input += torch.Tensor(np.random.normal(0, 128./255., size=temp_input.shape))
                perturb_data[perturb_data_idx] = temp_input
            print("perturb data shape: ", perturb_data.shape)


            print("Starting generate_data_batches: ", datetime.datetime.now(), flush=True)

            data_loader = self.generate_data_batches(perturb_data, orig_labels, orig_IDs, batch)
            num_imgs        = perturb_data.shape[0]
            del perturb_data
            
            # total_images = 0
            # for data_batch, label_batch, id_batch in data_loader:
            #     total_images += data_batch.shape[0]
            # print(f"Total images in data loader: {total_images}")

            print("Starting generate_activations perturbed: ", datetime.datetime.now(), flush=True)

            if act_perturb is None:
                act_perturb, _, _, self.sampledict = generate_activations(data_loader, self.model, device, layer, save_act=True, save_act_fname=save_act_fname, save_id_fname=save_id_fname, ret_act=False, f_idx=sens_idx, sampledict=self.sampledict, perturbed=True, trial=trial)
        
            else:
                act_perturb, _, _, self.sampledict = generate_activations(data_loader, self.model, device, layer, save_act=True, save_act_fname=save_act_fname, save_id_fname=save_id_fname, ret_act=False, mid_level=True, f_idx=sens_idx, sampledict=self.sampledict, perturbed=True, trial=trial)

        e_time = time.time()
        print('Time: %f'%(e_time - s_time))

        return num_imgs 


    ###################
    #### need an extra_loader here
    ###################

    # #### Generate mask for data samples (to apply curriculum) ####
    # def gen_data_mask(self, stats, sample_percentage):
    #     print('\n stat is' ,  stats.shape[0])

    #     # mask   = np.where(stats<=np.sort(stats)[::-1][int(sample_percentage*stats.shape[0])])[0]
    #     mask   = np.where(stats<=np.sort(stats)[::-1][int(sample_percentage)])[0]

    #     return mask 



    def gen_data_mask(self, stats, sample_percentage):
        # print('\n stat is' ,  stats.shape[0])
        labels = []
        IDs = []
        for data, target, ID in self.extraloader:
            # Append the labels to the list
            labels.extend(target.numpy())
            IDs.extend(ID)

        ### Note: Stats is a list of the mean perturbation magnitudes for all samples, with shape [# samples]

        # print("Ids length: ", len(IDs), " ", IDs, flush=True)

        ### Sorts in descending order
        sorted_indices =  np.argsort(stats)[::-1]

        # threshold=self.args.caperClassAllowance
        threshold=self.args.classRemovalAllowance
        
        totalRemoveCount = sample_percentage
        # top_indices = sorted_indices[:int(sample_percentage)]
        # See how many samples from each class are in this set


        mask = []
        removedCount = 0
        class_removed = {label: 0 for label in np.unique(labels)}  # Dictionary to count class frequencies
        for idx in sorted_indices:
            ### If we can still remove more of the given class, and havent removed the total amount allowed, then remove the idx
            if (class_removed[labels[idx]] < threshold) and (removedCount < totalRemoveCount):
                class_removed[labels[idx]] += 1
                removedCount += 1
                mask.extend([idx])
                self.sampledict['removed'][IDs[idx].item()] = 1


       
        # print("Mask: ", mask)
        # Now create the complement of the mask
        total_indices = set(range(len(stats)))  # Full set of indices
        print('\n mask len is', len(mask), 'set mask len is', len(set(mask)))
        mask_set = set(mask)  # Convert mask to set
        mask = np.array(list(total_indices - mask_set))  # Find the complement
        # print("Mask set subtract: ", mask)
        return mask





    
    

    def New_Data(self, Save_dir):

        # Load Data
        # trainloader, extraloader, testloader = data_loader(Dataset, Batch_size, val_load=True)

        # # Check if GPU is available (CUDA)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load Network
        if self.args.arch in ['vgg16', 'resnet18', 'resnet50', 'modresnet18']:
            
            weight_names= []
            for names, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    for param_name, param in module.named_parameters():
                        if "weight" in param_name:
                            # Filter only weights
                            full_param_name = f"{names}.{param_name}"  # Create the full parameter name
                            weight_names.append(full_param_name)
            
                    
            layers = {'layers': weight_names}
            # layers = np.load('vgg16_bn_cifar_dict.npy', allow_pickle=True).item()

            
        
        # elif self.args.arch == 'resnet50':
        #     layers = np.load('resnet50_v2_cifar_dict.npy', allow_pickle=True).item()

        else:
            print('Invalid Model Selection! Exiting')
            exit(1)

        # END IF
        collect_stats = None
        if self.args.Window == 'fhalf':
            layer_idxs = np.arange(0,int(len(layers['layers'])/2.),1)

        elif self.args.Window == 'shalf':
            layer_idxs = np.arange(len(layers['layers'])-1, int(len(layers['layers'])/2.)-1, -1)

        #!# This doesnt actually get the final layer only, its all layers up to it
        elif self.args.Window == 'final':
            if self.args.arch == 'vgg16':                
                layer_idxs = np.asarray([len(layers['layers'])-3-1])
            elif self.args.arch in ['resnet18', 'modresnet18', 'resnet50']:
                print("Using resnet layer")
                layer_idxs = np.asarray([len(layers['layers'])-1-1])   
                # layer_idxs = np.asarray([len(layers['layers'])-2-1])   
            else:
                layer_idxs = np.asarray([len(layers['layers'])-2-1])
            print("Final window layer idxs Caper: ", layer_idxs)


        else:
            layer_idxs = np.arange(0, len(layers['layers']), 1)


        if self.args.Window == 'gaussian':
            if len(layers['layers'])%2 == 1:
                multiplier = scipy.stats.norm(0,1).pdf(np.arange(-int(len(layers['layers'])/2.), int(len(layers['layers'])/2.+1), 1))
            else: 
                multiplier = scipy.stats.norm(0,1).pdf(np.arange(-int(len(layers['layers'])/2.), int(len(layers['layers'])/2.), 1))
        else:
            multiplier = np.ones(layer_idxs.shape)

        assert(multiplier.shape[0] == layer_idxs.shape[0])

        m_counter = 0





        for c_layer in layer_idxs:
            print('Processing: %s'%(layers['layers'][c_layer]))

            if 'fc' in layers['layers'][c_layer] or 'fc' in layers['layers'][c_layer+1]:
                continue


            logitsBestFile = Save_dir+'/logits_best_'+str(self.args.caper_epsilon)+'_'+ self.args.Window+'_'+\
                                                    str(self.args.sample_percentage)+'_'+str(self.args.tau)+'_'+layers['layers'][c_layer]+'.txt'
            idsBestFile = Save_dir+'/ids_'+str(self.args.caper_epsilon)+'_'+ self.args.Window+'_'+\
                                                    str(self.args.sample_percentage)+'_'+str(self.args.tau)+'_'+layers['layers'][c_layer]+'.txt'            
            
            if collect_stats is None:
                if 'shortcut' in layers['layers'][c_layer+1]:
                    if 'fc' in layers['layers'][c_layer+2]:
                        continue
                    print("Using shortcut generate_counts()")
                    num_imgs = self.generate_counts(self.model, self.extraloader, device, layers['layers'][c_layer], layers['layers'][c_layer+2], self.args.batch_size, 10, logitsBestFile, idsBestFile)
                else:
                    print("Using standard generate_counts()")                    
                    num_imgs = self.generate_counts(self.model, self.extraloader, device, layers['layers'][c_layer], layers['layers'][c_layer+1], self.args.batch_size, 10, logitsBestFile, idsBestFile)
                
                collect_stats = multiplier[m_counter]* self.generate_stats(logitsBestFile, idsBestFile, size=num_imgs)

            else:
                print("Running with collect stats not None")
                if 'shortcut' in layers['layers'][c_layer+1]:
                    if 'fc' in layers['layers'][c_layer+2]:
                        continue
                    
                    num_imgs = self.generate_counts(self.model, self.extraloader, device, layers['layers'][c_layer], layers['layers'][c_layer+2], self.args.batch_size, 10, logitsBestFile, idsBestFile)
                else:
                    num_imgs = self.generate_counts(self.model, self.extraloader, device, layers['layers'][c_layer], layers['layers'][c_layer+1], self.args.batch_size, 10, logitsBestFile, idsBestFile)
                
                collect_stats += multiplier[m_counter]* self.generate_stats(logitsBestFile, idsBestFile, size=num_imgs)

            # END IF

            m_counter += 1

        return self.gen_data_mask(collect_stats, self.args.sample_percentage)