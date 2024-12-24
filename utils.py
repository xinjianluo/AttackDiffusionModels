import torch
import random 
import numpy as np
import math
import time 

import torch.nn as nn
import torchvision
import random 
import torch.optim as optim

import torch.nn.functional as F


def resetRandomStates(manualseed=47, printSeed=False):
    if printSeed:
        logging.info("In resetRandomStates(), set random seed =", manualseed)
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)
    
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dm %ds' % (m, s) if h==0 else '%dh %dm %ds' % (h, m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs)) 