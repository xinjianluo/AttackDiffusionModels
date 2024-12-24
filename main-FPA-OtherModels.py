import configparser
from datetime import datetime
import os 
import torch
import numpy as np
import random
import logging

from datasets import FairFullDataset, getPredefDatasetParam, PoisoningDataset, ConditionalPoisoningDataset
from MIestimation import extractClipFeat, trainMINE
from diffusion import samplingFromDDPM, trainDDPM, samplingConditionalModels, trainConditionalModels
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
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    modelname = "RHVAE"  # "NCSN", "SDE", "SSGAN", "SNGAN"
    samplenum = 6000
    poisoningAlpha = 0.5
    # attackType = "LabelFlipping"    # MISample, LabelFlipping, NoAttack
    datasetname = "CelebA-A"   # CelebA-S, CelebA-A
    class2Train = "01"
    
    # init logging
    logfile = f'Fairness_{modelname}_{datasetname}_{samplenum}_{poisoningAlpha}_class{class2Train}_{getTimeStamp()}.log'
    logpath = currentDir() + os.sep + "log" + os.sep + logfile
    initlogging(logpath)
    logging.critical(f"Writting to log file {logfile} ...")
    
    logging.critical(f"Running on device {device}")
    
    for datasetname in ("CelebA-A", "CelebA-S"):
    
        # load dataset 
        img_channels, img_size, img_class, transform = getPredefDatasetParam(datasetname)
     
      
        logging.critical(f"Dataset: {datasetname}; Classes: {img_class}; Channels: {img_channels}; Image Size: {img_size}")
        
        for modelname in ["RHVAE"]:   # "SSGAN", "SNGAN", "NCSN", "SDE"
            mainStart = time.time()
            atypes = ["NoAttack", "MISample", "LabelFlipping"]
            for i, attackType in enumerate(atypes):
                logging.critical("\n\n")
                logging.critical(f"Start processing attackType {attackType} for dataset {datasetname} and model {modelname}")

                if attackType == "LabelFlipping":
                    pX = torch.load(f"samplingdata/X_{datasetname}_6000_0.5_Guided-DDPM_class-1_LabelFlipping.arr")
                    pY = torch.load(f"samplingdata/Y_{datasetname}_6000_0.5_Guided-DDPM_class-1_LabelFlipping.arr")
                elif attackType == "MISample":
                    pX = torch.load(f"samplingdata/X_{datasetname}_6000_0.5_Guided-DDPM_class-1_MISample.arr")
                    pY = torch.load(f"samplingdata/Y_{datasetname}_6000_0.5_Guided-DDPM_class-1_MISample.arr")
                elif attackType == "NoAttack":
                    pX = torch.load(f"samplingdata/X_{datasetname}_6000_0_Guided-DDPM_class-1_MISample.arr")
                    pY = torch.load(f"samplingdata/Y_{datasetname}_6000_0_Guided-DDPM_class-1_MISample.arr")
                
                # diffusion training dataset 
                diffTrainData = ConditionalPoisoningDataset(pX, pY) 
                 
                # train diffusion model 
                savemodpath = f"models/ConditionalModel_{datasetname}_{samplenum}_{poisoningAlpha}_{modelname}_class{class2Train}_{attackType}.pt"
                # NCSN 1500 epochs, GANs 15000 steps
                if modelname == "NCSN":
                    n_epochs = 1500
                elif modelname in ("SSGAN", "SNGAN"):
                    n_epochs = 15000
                elif modelname == "SDE":
                    n_epochs = None 
                elif modelname == "RHVAE":
                    n_epochs = 10
                else:
                    assert False, f"Unsupported modelname {modelname}"
                assert n_epochs is not None, f"Please set training epochs for dataset {datasetname}"
                diffusionModel = trainConditionalModels(diffTrainData, datasetname, modelname, img_channels, img_size, n_epochs, device, savemodpath)
                
                # sampling classification data 10000
                dtX, dtY = samplingConditionalModels(diffusionModel, modelname, datasetname, img_channels, img_size, device, sampleNum=6000)
                savesamplepath = f"samplingdata/Samples_ConditionalModel_{datasetname}_10000_{poisoningAlpha}_{modelname}_class{class2Train}_{attackType}.arr"
                torch.save(dtX, savesamplepath)
                
                classificationTrainData = PoisoningDataset(dtX, dtY)
                classificationTestData = FairFullDataset(split="test", datasetname=datasetname, transform=transform, sampleNum=10000)
                
                trainDownStreamClassifier(classificationTrainData, classificationTestData, device)
                
                logging.critical(f"Finished processing attackType {attackType} for dataset {datasetname} and model {modelname}")
                logging.critical(f"time passed: {timeSince(mainStart, float(i+1)/len(atypes))}") 
    
    
    logging.critical("\n\n<----------------- ALL Finished ----------------->\n\n")
    
    