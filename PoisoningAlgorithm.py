from npeet import entropy_estimators as ee
import logging
import torch
import numpy as np 
import time 

from utils import resetRandomStates, timeSince

import torch
import torch.nn as nn
import torchvision
import random 
import torch.optim as optim

import math
import numpy as np
from torch.utils.data import Dataset

import torch.nn.functional as F

class GreedySampling():

    def __init__(self, X, Xigs, S, Y, mine):

        self.mi_constant = mine.mi(X, Y.unsqueeze(1))
        self.delta = self.mi_constant * 0.1
        logging.critical("Preprocessing in GreedySampling...")
        logging.info(f"The contrainted target MI: {self.mi_constant}")
        logging.info(f"Delta: {self.delta}")
        logging.info(f"X.shape={X.shape}, S.shape={S.shape}, Y.shape={Y.shape}")
        self.X = X
        self.S = S 
        self.Y = Y 
        self.Xigs = Xigs 
        self.mine = mine 
        
    def printMI(self, baseX, baseS, baseY, title):
        logging.info(title)
        sy_mi = ee.midd(baseS.numpy(), baseY.numpy())
        logging.info(f"MI between S and Y is {sy_mi}")
        with torch.no_grad():
            xy_mi = self.mine.mi(baseX, baseY.unsqueeze(1)).detach().cpu()
        lower = self.mi_constant - self.delta 
        upper = self.mi_constant + self.delta
        logging.info(f"MI between X and Y is {xy_mi}, is in ({lower}, {upper})? {lower<=xy_mi<=upper}")

    def maxSY(self, baseS, baseY):
        if type(baseS) == torch.Tensor: 
            baseS = baseS.numpy() 
        if type(baseY) == torch.Tensor:
            baseY = baseY.numpy()
        results = dict() 
        baseMI = ee.midd(baseS, baseY)
        for s in (0, 1):
            for y in (0, 1):
                augMI = ee.midd(np.concatenate((baseS, [s])), np.concatenate((baseY, [y])))
                aug = augMI - baseMI
                results[aug] = "s{}y{}".format(s, y)
                
        tuple_list = sorted(results.items(), reverse=True)
        return tuple_list 

    def getBestIdxWithinConstraint(self, set2use, baseX, baseY):
        augidx_dict = dict() 
        best_aug = 1e10
        bestidx = -1 
        sampleNum = 100
        if len(set2use) < sampleNum:
            subset2use = set2use
        else:
            subset2use = np.random.choice(list(set2use), size=sampleNum, replace=False)
        
        for idx in subset2use:
            newx = self.X[idx:idx+1]
            newy = self.Y[idx:idx+1]
            newXarr = torch.cat((baseX, newx))
            newYarr = torch.cat((baseY, newy))
            with torch.no_grad():
                newMI = self.mine.mi(newXarr, newYarr.unsqueeze(1)).detach().cpu()
            miaug = abs(newMI - self.mi_constant)
            if miaug < best_aug:
                best_aug = miaug
                bestidx = idx 
        success = True if best_aug < self.delta else False
        assert bestidx > -1, "bestidx abnormal"
        return bestidx, success
        
    def printStatistics(self, target_idx_arr):
        Ssamples = self.S[target_idx_arr]
        Ysamples = self.Y[target_idx_arr]
        s0y0idxset = set()
        s1y0idxset = set()
        s0y1idxset = set()
        s1y1idxset = set()
        for idx in range(len(Ssamples)):
            if Ssamples[idx] == 0 and Ysamples[idx] == 0:
                s0y0idxset.add(idx)
            elif Ssamples[idx] == 0 and Ysamples[idx] == 1:
                s0y1idxset.add(idx) 
            elif Ssamples[idx] == 1 and Ysamples[idx] == 0:
                s1y0idxset.add(idx) 
            elif Ssamples[idx] == 1 and Ysamples[idx] == 1:
                s1y1idxset.add(idx) 
        
        logging.info(f"For class 0, {len(s0y0idxset) + len(s1y0idxset)} -> [{len(s0y0idxset)}, {len(s1y0idxset)}]")
        logging.info(f"For class 1, {len(s0y1idxset) + len(s1y1idxset)} -> [{len(s0y1idxset)}, {len(s1y1idxset)}]")

    def startSampling(self, targetcard=6000, alpha=0.5):
        logging.critical("Start the Greedy Sampling Algorithm...")
        resetRandomStates()

        # initialization
        basecard = int(targetcard*(1-alpha)) 
        logging.info(f"targetcard={targetcard}, basecard={basecard}")

        # initialize
        target_idx_arr = torch.zeros(targetcard)

        while True:
            base_idx_arr = np.random.choice(range(len(self.X)), size=basecard, replace=False)
        #     mi_curr = ee.micd(X[base_idx_arr].numpy(), Y[base_idx_arr].numpy().reshape(basecard, 1))
            mi_curr = self.mine.mi(self.X[base_idx_arr], self.Y[base_idx_arr].unsqueeze(1))
            # logging.info(mi_curr)
            if abs(self.mi_constant - mi_curr) < self.delta:
                break
                

        base_idx_set = set(base_idx_arr)

        # assign base values
        curr_len = basecard 
        target_idx_arr[:curr_len] = torch.from_numpy(base_idx_arr)
        logging.info("Initialization Finished!")

        # define different sets
        unused_idx_set = set(range(len(self.X))) - base_idx_set
        unused_s0y0_set = set() 
        unused_s0y1_set = set()
        unused_s1y0_set = set()
        unused_s1y1_set = set()

        # initialize sets
        for idx in unused_idx_set:
            if self.S[idx] == 0 and self.Y[idx] == 0:
                unused_s0y0_set.add(idx)
            elif self.S[idx] == 0 and self.Y[idx] == 1:
                unused_s0y1_set.add(idx)
            elif self.S[idx] == 1 and self.Y[idx] == 0:
                unused_s1y0_set.add(idx)
            elif self.S[idx] == 1 and self.Y[idx] == 1:
                unused_s1y1_set.add(idx)

        SYset_dict = dict()
        SYset_dict["s0y0"] = unused_s0y0_set
        SYset_dict["s0y1"] = unused_s0y1_set
        SYset_dict["s1y0"] = unused_s1y0_set
        SYset_dict["s1y1"] = unused_s1y1_set

        target_idx_arr = target_idx_arr.long()

        self.printMI(self.X[target_idx_arr[:curr_len]], self.S[target_idx_arr[:curr_len]], self.Y[target_idx_arr[:curr_len]], "The base samples:")

        starttime = time.time()

        for newidx in range(basecard, targetcard):     
            assert curr_len == newidx, "curr_len abnormal"
            # first, maximum the MI between S and Y 
            tuple_list = self.maxSY(self.S[target_idx_arr[:curr_len]], self.Y[target_idx_arr[:curr_len]])
            success_flag = False
            # second, for each (S, Y) tuple, compute the MI augmentation in the corresponding set
            for _, combo in tuple_list:
                assert combo in ("s0y0", "s1y0", "s0y1", "s1y1"), "combo abnormal"
                set2use = SYset_dict[combo]
                
                # third, for all items in the set, compute MI augmentation and sort
                bestidx, flag = self.getBestIdxWithinConstraint(set2use, self.X[target_idx_arr[:curr_len]], self.Y[target_idx_arr[:curr_len]])
                if flag == False:
                    continue
                else:
                    success_flag = True
                    target_idx_arr[curr_len] = bestidx
                    curr_len += 1
                    # remove the idx from the set
                    set2use.remove(bestidx)
                    break 
        #     assert success_flag == True, f"Failed to find a suitable sample at idx {newidx}"
            if success_flag == False:
                target_idx_arr[curr_len] = bestidx
                curr_len += 1
                # remove the idx from the set
                set2use.remove(bestidx)
            
            if newidx % ((targetcard-basecard) // 4) == 0:
                logging.critical("-------------------------------")
                logging.critical(f"Completed sampling record {newidx}, {targetcard-newidx} to go")
                # time computation
                logging.critical("time passed: {}".format(timeSince(starttime, float(newidx-basecard+1)/(targetcard-basecard))))
                self.printMI(self.X[target_idx_arr[:curr_len]], self.S[target_idx_arr[:curr_len]], self.Y[target_idx_arr[:curr_len]], "In generation:")
                
        assert (target_idx_arr == 0).sum() <= 1, "target_idx_arr element abnormal"
        assert curr_len == targetcard, "curr_len != targetcard"

        self.printMI(self.X[target_idx_arr[:curr_len]], self.S[target_idx_arr[:curr_len]], self.Y[target_idx_arr[:curr_len]], "The final samples:")
        self.printStatistics(target_idx_arr)
        logging.critical("Greedy Sampling finished!")   
        return self.Xigs[target_idx_arr[:curr_len]], self.S[target_idx_arr[:curr_len]], self.Y[target_idx_arr[:curr_len]]
    
    
    
class LabelFlipping():

    def __init__(self, Xigs, S, Y):
        logging.critical("Preprocessing in LabelFlipping...")
        logging.info(f"Xigs.shape={Xigs.shape}, S.shape={S.shape}, Y.shape={Y.shape}")
        self.Xigs = Xigs
        self.S = S 
        self.Y = Y 
   
    def printStatistics(self, Ssamples, Ysamples):
        # Ssamples = self.S[target_idx_arr]
        # Ysamples = self.Y[target_idx_arr]
        s0y0idxset = set()
        s1y0idxset = set()
        s0y1idxset = set()
        s1y1idxset = set()
        for idx in range(len(Ssamples)):
            if Ssamples[idx] == 0 and Ysamples[idx] == 0:
                s0y0idxset.add(idx)
            elif Ssamples[idx] == 0 and Ysamples[idx] == 1:
                s0y1idxset.add(idx) 
            elif Ssamples[idx] == 1 and Ysamples[idx] == 0:
                s1y0idxset.add(idx) 
            elif Ssamples[idx] == 1 and Ysamples[idx] == 1:
                s1y1idxset.add(idx) 
        
        logging.info(f"For class 0, {len(s0y0idxset) + len(s1y0idxset)} -> [{len(s0y0idxset)}, {len(s1y0idxset)}]")
        logging.info(f"For class 1, {len(s0y1idxset) + len(s1y1idxset)} -> [{len(s0y1idxset)}, {len(s1y1idxset)}]")

    def startSampling(self, targetcard=6000, alpha=0.5):
        logging.critical("Start the Label Flipping Algorithm...")
        resetRandomStates()

        init_idx_arr = np.random.choice(range(len(self.Xigs)), size=targetcard, replace=False)
        initS = self.S[init_idx_arr]
        initY = self.Y[init_idx_arr]
        for i in range(int(targetcard*(1-alpha)), targetcard):
            if initS[i] == 0: 
                initY[i] = 0 
            elif initS[i] == 1:
                initY[i] = 1 

        self.printStatistics(initS, initY)
        logging.critical("Label Flipping finished!")   
        return self.Xigs[init_idx_arr], initS, initY

