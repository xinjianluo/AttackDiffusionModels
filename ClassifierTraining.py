import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import Dataset
import logging  
from utils import resetRandomStates

import random 
import torch.optim as optim


import math
import time 

import torch.nn.functional as F



class BinaryClassifier(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, n_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def check_fairness_gap(model, dataset, device):
    abase = 0.0
    aprimebase = 0.0
    apositive = 0.0 
    aprimpositive = 0.0
    model.eval()
    for x, s, y in dataset:
        with torch.no_grad():
            yhat = model(x.unsqueeze(0).to(device))
            yhat = yhat.argmax()
        if s == 1:
            abase += 1 
            if yhat == 1:
                apositive += 1
        elif s == 0:
            aprimebase += 1
            if yhat == 1:
                aprimpositive += 1 
    gap = abs(apositive / abase - aprimpositive / aprimebase)
    return gap


def check_test_accuracy(mymodel, dataloader, device):
    mymodel.eval()
    accur = 0.0
    base = 0
    with torch.no_grad():
        for x, _, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            yhat = mymodel(x)
            accur += ( (yhat.argmax(dim=1)) == y ).sum()
            base += x.shape[0]
    return accur / base

def trainDownStreamClassifier(traindata, testdata, device):
    """
        traindata -> (x, y)
        testdata  -> (x, s, y)
    """
    resetRandomStates()
    model = BinaryClassifier(n_channels=3, n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = 40
    test_interval = int(epochs / 4)
    
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=64, shuffle=True, num_workers=4)

    for epoch in range(1, epochs+1):
        accurate = 0.0
        train_accur_base = 0.0
        for x, y in trainloader: 
            model.train()
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y.long())
            loss.backward()
            optimizer.step()
            accurate += ((yhat.argmax(dim=1))==y).sum()  
            train_accur_base += x.shape[0]

        if epoch % test_interval == 0 or epoch == epochs - 1:
            # for each epoch, print information
            train = accurate / train_accur_base
            test = check_test_accuracy(model, testloader, device)
            logging.critical(f"In epoch {epoch} / {epochs}, train accur is {train}, test accur is {test}.")


    logging.critical(f"Fairness GAP is {check_fairness_gap(model, testdata, device)}")
    logging.critical("All finished!")  