import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import Dataset


from utils import resetRandomStates

import random 
import torch.optim as optim

import math
import time 


import torch.nn.functional as F



def getPredefDatasetParam(datasetname):
    if datasetname in ("MNIST", "FashionMNIST"):
        img_channels = 1 
        img_size = 32
        img_class = 10 
    elif datasetname in ("CelebA-A", "CelebA-S"):
        img_channels = 3 
        img_size = 32 
        img_class = 2
    elif datasetname == "CIFAR10":
        img_channels = 3 
        img_size = 32 
        img_class = 10 

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor()
    ])
    
    return img_channels, img_size, img_class, transform

def processCelebA(split, transform, datasetname, sampleNum=30000):
    resetRandomStates()

    fulldataset = torchvision.datasets.CelebA(root="../datasets", split=split, download=False, transform=transform)
   
    fullloader = torch.utils.data.DataLoader(fulldataset, batch_size = len(fulldataset), shuffle=False)
    fullimgs, fulllabels = next(iter(fullloader))
    
    sampleSize = sampleNum
    if len(fullimgs)>sampleSize:
        rand_idx_arr = np.random.choice(range(len(fullimgs)), size=sampleSize, replace=False)
        fullimgs = fullimgs[rand_idx_arr]
        fulllabels = fulllabels[rand_idx_arr]
    
    malelabels = fulllabels[:, 20].squeeze()
    assert datasetname in ("CelebA-S", "CelebA-A"), f"datasetname={datasetname} abnormal"
    if datasetname == "CelebA-S":    # smiling 
        targetlabels = fulllabels[:, 31].squeeze()
    elif datasetname == "CelebA-A":    # attractive
        targetlabels = fulllabels[:, 2].squeeze()
    
    s0y0idxset = set()
    s1y0idxset = set()
    s0y1idxset = set()
    s1y1idxset = set()
    for idx in range(len(fullimgs)):
        if malelabels[idx] == 0 and targetlabels[idx] == 0:
            s0y0idxset.add(idx)
        elif malelabels[idx] == 0 and targetlabels[idx] == 1:
            s0y1idxset.add(idx) 
        elif malelabels[idx] == 1 and targetlabels[idx] == 0:
            s1y0idxset.add(idx) 
        elif malelabels[idx] == 1 and targetlabels[idx] == 1:
            s1y1idxset.add(idx) 
    fairportion = np.min((len(s0y0idxset), len(s0y1idxset), len(s1y0idxset), len(s1y1idxset)))
    fairidx = set(np.random.choice(list(s0y0idxset), size=fairportion, replace=False)) \
            | set(np.random.choice(list(s0y1idxset), size=fairportion, replace=False)) \
            | set(np.random.choice(list(s1y0idxset), size=fairportion, replace=False)) \
            | set(np.random.choice(list(s1y1idxset), size=fairportion, replace=False))
    
    fairidx = torch.tensor(list(fairidx), dtype=torch.long)  
    assert len(fairidx) == fairportion*4, "len(fairidx) abnormal"
    fullimgs = fullimgs[fairidx]
    malelabels = malelabels[fairidx]
    targetlabels = targetlabels[fairidx]
    return fullimgs, malelabels, targetlabels



class FairFullDataset(Dataset):

    def __init__(self, split="test", datasetname="CelebA-S", transform=None, sampleNum=30000):
        """
            split (string): One of {‘train’, ‘test’}
        """
        self.fullimgs, self.malelabels, self.targetlabels = None, None, None 
        if datasetname in ("CelebA-S", "CelebA-A"):
            self.fullimgs, self.malelabels, self.targetlabels = processCelebA(split, transform, datasetname, sampleNum)
        
       
    def __len__(self):
        return len(self.fullimgs)
    
    def __getitem__(self, index):
        return self.fullimgs[index], self.malelabels[index], self.targetlabels[index] 
        
class PoisoningDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]                 
        
        
        
class ConditionalPoisoningDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.label = None 
        
    def resetClass(self, label):
        self.label = label
        self.XC = self.X[self.Y == label]
        self.YC = self.Y[self.Y == label]
        
    def __len__(self):
        assert self.label is not None, "In ConditionalPoisoningDataset(), did not run resetClass()"
        return len(self.XC)
    
    def __getitem__(self, index):
        assert self.label is not None, "In ConditionalPoisoningDataset(), did not run resetClass()"
        return self.XC[index], self.YC[index]         
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        