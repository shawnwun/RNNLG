######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import numpy as np
import math

def softmax(w):
    e = np.exp(w)
    dist = e/np.sum(e)
    return dist

def sigmoid(w):
    e = np.exp(-w)
    acti = 1/(1+e)
    return acti

def tanh(w):
    return np.tanh(w)


