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
from AuxiliaryScripts.Normalization_metrics import Normalization_Techniques
import os
from AuxiliaryScripts import corruptions 
import datetime

class EpochAcc_Method():
    
    def __init__(self, args, model, extra_loader, sampledict):
        self.args = args
        self.model = model
        self.extraloader= extra_loader
        self.sampledict = sampledict




    def gen_data_mask(self):
        # print('\n stat is' ,  stats.shape[0])
        labels = []
        IDs = []
        for data, target, ID in self.extraloader:
            # Append the labels to the list
            labels.extend(target.numpy())
            IDs.extend(ID)






        predsDict = self.sampledict['epochlogits']
        labelsDict = self.sampledict['labels']
        accsDict = {}

        loss = nn.CrossEntropyLoss()

        if self.args.EpochAccEpochs == -1:
            startEpoch = 0
        else:
            startEpoch = self.args.tau - self.args.EpochAccEpochs

        # for ID in labelsDict.keys():
        for ID in range(len(list(labelsDict.keys()))):
            accsDict[ID] = []
            cumulativePerformance, totalCount = 0, 0


            ### For each epoch of training up until epoch tau, record the cumulative performance on each training sample
            for e in range(startEpoch, self.args.tau, self.args.EpochAccInterval):
                if self.args.EpochAccMetric == "loss":
                    cumulativePerformance += (loss(predsDict[ID][e], labelsDict[ID]))
                else:
                    if torch.argmax(predsDict[ID][e]).item() == labelsDict[ID].item():
                        cumulativePerformance += 1

                totalCount += 1
                accsDict[ID].append((cumulativePerformance/totalCount)*100)





        allAccs = []
        reportingEpoch = self.args.tau - 1

        for ID in range(len(list(accsDict.keys()))):
            # allAccs.append(accsDict[ID][reportingEpoch])
            ### Always report the cumulative avg loss on the most recent epoch
            allAccs.append(accsDict[ID][-1])

        print("Length of allAccs: ", len(allAccs))
    
        allAccs = torch.tensor(allAccs)



        #!# Ascending works best for Softmax removal, Descending works best for Loss removal
        if self.args.sortOrder == "ascending":
            sorted_indices = torch.argsort(allAccs)
        elif self.args.sortOrder == "descending":
            sorted_indices = torch.argsort(allAccs, descending=True)

        ### These are the training sample indices to be removed based on the tracked metric (cumulative loss or accuracy) over the e<tau epochs of training
        sorted_indices = sorted_indices.cpu().numpy()


        threshold=self.args.classRemovalAllowance
        
        totalRemoveCount = self.args.sample_percentage

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
        total_indices = set(range(len(list(accsDict.keys()))))  # Full set of indices
        print('\n mask len is', len(mask), 'set mask len is', len(set(mask)))
        mask_set = set(mask)  # Convert mask to set
        mask = np.array(list(total_indices - mask_set))  # Find the complement
        # print("Mask set subtract: ", mask)
        return mask





    