import numpy as np
import torch
import csv
import copy
#import scipy.stats as stats
import scipy.stats
from matplotlib import pyplot as plt
from operator import itemgetter
import random

def gpu(x):
    '''Transforms numpy array or torch tensor it torch.cuda.FloatTensor'''
    if isinstance(x, np.ndarray):
        return torch.cuda.FloatTensor(x.astype('float32'))
    else:
        return torch.cuda.FloatTensor(x)


def cpu(x):
    '''Transforms torch tensor into numpy array'''
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.cpu().detach().numpy()



def sigmoid(x):
    '''Returns sigmoid(x)'''
    return 1 / (1 + np.exp(-x))



def flip_filt(filt):
    '''Returns filter flipped over x and y dimension'''
    return np.ascontiguousarray(filt[...,::-1,::-1])
