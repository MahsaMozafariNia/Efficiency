"""Simple data generator for training and testing data"""

import numpy as np
import scipy.io as spio
import math
import collections

from torch.utils.data import Dataset
from PIL import Image

### Generic data generator for all datasets
class SimpleDataGenerator(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
    def __len__(self):
        """Denotes the total number of examples, later index will be sampled according to this number"""
        return int(len(self.x))
    
    def __getitem__(self, index):
        img, target = self.x[index], self.y[index]

        ### Transforms were done manually prior to calling the DataGenerator, so self.transform should be None
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        
        return img, target



class IdTrackDataGenerator(Dataset):
    def __init__(self, x, y, z, transform=None):
        self.x = x
        self.y = y
        self.z = z
        self.transform = transform
        
    def __len__(self):
        """Denotes the total number of examples, later index will be sampled according to this number"""
        return int(len(self.x))
    
    def __getitem__(self, index):
        img, target, ID = self.x[index], self.y[index], self.z[index]

        ### Transforms were done manually prior to calling the DataGenerator, so self.transform should be None
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        
        return img, target, ID



