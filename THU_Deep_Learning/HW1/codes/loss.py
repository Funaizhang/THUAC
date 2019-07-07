from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        loss_arr = np.square(target - input)
        loss = np.sum(loss_arr)/target.shape[0]/2
        return loss

    def backward(self, input, target):
        
        assert input.shape == target.shape,"EuclideanLoss.backward arguments input and target not the same shape"
        
        grad = (input - target)/target.shape[0]
        return grad


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        
        assert input.shape == target.shape,"SoftmaxCrossEntropyLoss.forward arguments input and target not the same shape"
        
        # get unnormalized probabilities
        unnormalized_p = np.exp(input)
        # normalize them for each example
        p = unnormalized_p / np.sum(unnormalized_p, axis=1, keepdims=True)
        
        assert p.shape == target.shape,"SoftmaxCrossEntropyLoss.forward shape wrong"
        
        loss_arr = np.multiply(np.log(p), target)   
        loss = - np.sum(loss_arr)/target.shape[0]
        return loss
        
    
    def backward(self, input, target):
        
        # get unnormalized probabilities
        unnormalized_p = np.exp(input)
        # normalize them for each example
        p = unnormalized_p / np.sum(unnormalized_p, axis=1, keepdims=True)
        
        assert input.shape == p.shape,"SoftmaxCrossEntropyLoss.backward wrong"
        
        grad = (p - target)/target.shape[0]
        return grad