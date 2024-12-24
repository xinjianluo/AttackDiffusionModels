import torch
import torch.nn as nn
import torchvision
import clip
import logging
import time 
import math 
import numpy as np

import random 
import torch.optim as optim

from utils import resetRandomStates, timeSince

from torch.utils.data import Dataset


import torch.nn.functional as F


def extractClipFeat(samples, device):
    model, _ = clip.load("ViT-B/32", device=device)
    ttlen = len(samples)
    clipfeatures = torch.zeros(ttlen, 512)

    with torch.no_grad():    
        for startidx in range(0, ttlen, 1000):
            endidx = startidx + 1000 
            if endidx > ttlen:
                endidx = ttlen 

            with torch.no_grad():
                image_features = model.encode_image(torchvision.transforms.Resize(224)(samples[startidx:endidx].to(device)) )
            clipfeatures[startidx:endidx] = image_features.detach().cpu() 
        
    logging.critical("CLIP feature extraction all done!")
    return clipfeatures
   
torch.autograd.set_detect_anomaly(True)

EPS = 1e-6

def batch(x, y, batch_size=1, shuffle=True):
    assert len(x) == len(y), "Input and target data must contain same number of elements"
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    n = len(x)

    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm]
        y = y[rand_perm]

    batches = []
    for i in range(n // batch_size):
        x_b = x[i * batch_size: (i + 1) * batch_size]
        y_b = y[i * batch_size: (i + 1) * batch_size]

        batches.append((x_b, y_b))
    return batches


class ConcatLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x, y), self.dim)


class CustomSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input 
        

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean


class Mine(nn.Module):
    def __init__(self, T, preprocess=None, loss='mine', alpha=0.01, method=None, device=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.lossRecords = []
        self.alpha = alpha
        self.method = method
        self.preprocess = preprocess
        self.device = device
        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = CustomSequential(ConcatLayer(), *T)
            else:
                self.T = CustomSequential(ConcatLayer(), T)
        else:
            self.T = T
            
    def lossCurve(self):
        plt.figure() 
        plt.plot(self.lossRecords)

    def forward(self, x, z, z_marg=None):
        x = x.to(self.device)
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]
        if self.preprocess is not None:
            x = self.preprocess(x)
        
        z = z.to(self.device)
        z_marg = z_marg.to(self.device)
        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        resetRandomStates()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, epochs, batch_size=64, opt=None):

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        starttime = time.time()
        
        for iter in range(1, epochs + 1):
            mu_mi = 0
            # can change batch function to dataloader
            for x, y in batch(X, Y, batch_size):
                opt.zero_grad()
  
                loss = self.forward(x, y)
                loss.backward() 
                opt.step()

                mu_mi -= loss.item()
            self.lossRecords.append(loss.detach().item())
            if iter % (epochs // 4) == 0:
                logging.critical(f"It {iter} / {epochs} - MI: {mu_mi / batch_size}")
                logging.critical("time passed: {}".format(timeSince(starttime, float(iter)/epochs)))        

        final_mi = self.mi(X, Y)
        logging.critical(f"Final MI: {final_mi}")
        return final_mi
        
def trainMINE(X, Y, device, epochs=1000):
    logging.critical("Training MINE for estimating mutual information...")
    statistics_network = nn.Sequential(
        nn.Linear(512 + 1, 100),
        nn.ReLU(),
        nn.Linear(100, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
    )
    
    mine = Mine(
        T = statistics_network.to(device),
        preprocess=None,
        loss = 'mine', #mine_biased, fdiv
        method = 'concat',
        device = device
    )
    
    mine.optimize(X, Y.unsqueeze(1), epochs = epochs)
    return mine 



