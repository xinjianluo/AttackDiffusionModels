import configparser
from datetime import datetime
import os 
import torch
import numpy as np
import random
import logging

from datasets import FairFullDataset, getPredefDatasetParam, PoisoningDataset
from MIestimation import extractClipFeat, trainMINE
from diffusion import samplingFromDDPM, trainDDPM
from ClassifierTraining import trainDownStreamClassifier

from PoisoningAlgorithm import GreedySampling, LabelFlipping
from utils import resetRandomStates, timeSince

import torch.nn as nn
import torchvision
import random 
import torch.optim as optim



import math
import time 

import numpy as np
from torch.utils.data import Dataset

import torch.nn.functional as F

from functools import partial


def getTimeStamp():
    return datetime.now().strftime("%Y%m%d_%H_%M_%S")
    
def currentDir():
    return os.path.dirname(os.path.realpath(__file__))
    
    
def initlogging(logfile):
    # debug, info, warning, error, critical
    # set up logging to file
    logging.shutdown()
    
    logger = logging.getLogger()
    logger.handlers = []
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=logfile,
                        filemode='w')
    
    # create console handler
    ch = logging.StreamHandler()
    # Sets the threshold for this handler. 
    # Logging messages which are less severe than this level will be ignored, i.e.,
    # logging messages with < critical levels will not be printed on screen
    # E.g., 
    # logging.info("This should be only in file") 
    # logging.critical("This shoud be in both file and console")
    
    ch.setLevel(logging.CRITICAL)
    # add formatter to ch
    ch.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(ch)  


if __name__=='__main__':  
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    
    modelname = "Guided-DDPM"
    samplenum = 6000
    poisoningAlpha = 0.3
    attackType = "LabelFlipping"    # MISample, LabelFlipping
    datasetname = "CelebA-S"   # CelebA-S, CelebA-A
    class2Train = -1
    
    # init logging
    logfile = f'Fairness_{datasetname}_{samplenum}_{poisoningAlpha}_{modelname}_class{class2Train}_{attackType}_{getTimeStamp()}.log'
    logpath = currentDir() + os.sep + "log" + os.sep + logfile
    initlogging(logpath)
    logging.critical(f"Writting to log file {logfile} ...")
    
    logging.critical(f"Running on device {device}")
    
    # load dataset 
    img_channels, img_size, img_class, transform = getPredefDatasetParam(datasetname)
    if datasetname == "CelebA-A":
        partialdataset = FairFullDataset(split="train", datasetname=datasetname, transform=transform, sampleNum=60000)
    elif datasetname == "CelebA-S":
        partialdataset = FairFullDataset(split="train", datasetname=datasetname, transform=transform, sampleNum=30000)
  
    logging.critical(f"Dataset: {datasetname}; Classes: {img_class}; Channels: {img_channels}; Image Size: {img_size}")
    logging.critical(f"Train Dataset length: {len(partialdataset)}; Male Images: {(partialdataset.malelabels).sum()}; Target Positive Images: {(partialdataset.targetlabels).sum()}") 
    
    Xigs = partialdataset.fullimgs
    S = partialdataset.malelabels        # torch.Tensor
    Y = partialdataset.targetlabels      # torch.Tensor
        
    # preprocess the fullimages via CLIP
    clipfeatures = extractClipFeat(partialdataset.fullimgs, device)
        
    if attackType == "MISample":
        X = clipfeatures
        mine = trainMINE(X, Y, device, epochs=1000)    # DDPM, train 1000 epochs for MINE
    
    mainStart = time.time()
    alphalst = [0.1, 0.2, 0.3, 0.4, 0.5]
    for i, poisoningAlpha in enumerate(alphalst):
        logging.critical("\n\n")
        logging.critical(f"Start processing alpha {poisoningAlpha} for dataset {datasetname}")
        if attackType == "MISample":
            # sampling poisoning dataset 
            pX, pS, pY = GreedySampling(X, Xigs, S, Y, mine).startSampling(targetcard=6000, alpha=poisoningAlpha)
        elif attackType == "LabelFlipping":
            pX, pS, pY = LabelFlipping(Xigs, S, Y).startSampling(targetcard=6000, alpha=poisoningAlpha)
            
        torch.save(pX, f"samplingdata/X_{datasetname}_{samplenum}_{poisoningAlpha}_{modelname}_class{class2Train}_{attackType}.arr")
        torch.save(pS, f"samplingdata/S_{datasetname}_{samplenum}_{poisoningAlpha}_{modelname}_class{class2Train}_{attackType}.arr")
        torch.save(pY, f"samplingdata/Y_{datasetname}_{samplenum}_{poisoningAlpha}_{modelname}_class{class2Train}_{attackType}.arr")
        
        # diffusion training dataset 
        diffTrainData = PoisoningDataset(pX, pY)
        
        # train diffusion model 
        savemodpath = f"models/Model_{datasetname}_{samplenum}_{poisoningAlpha}_{modelname}_class{class2Train}_{attackType}.pt"
        diffusionModel = trainDDPM(diffTrainData, img_class, img_size, 800, device, savemodpath)    # DDPM, train 800 epochs for CelebA 
        
        # sampling classification data 
        dtX, dtY = samplingFromDDPM(diffusionModel, sampleNum=10000, device=device)
        classificationTrainData = PoisoningDataset(dtX, dtY)
        classificationTestData = FairFullDataset(split="test", datasetname=datasetname, transform=transform, sampleNum=10000)
        
        trainDownStreamClassifier(classificationTrainData, classificationTestData, device)
        
        logging.critical(f"Finished proprocessing alpha {poisoningAlpha} for dataset {datasetname}")
        logging.critical(f"time passed: {timeSince(mainStart, float(i+1)/len(alphalst))}") 
    
    
    logging.critical("\n\n<----------------- ALL Finished ----------------->\n\n")
    
    