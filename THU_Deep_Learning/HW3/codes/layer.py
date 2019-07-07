import torch
import torch.nn as nn
import numpy as np


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            '''Your codes here'''
            # x: shape = N x num_features
            x_mean = x.mean(dim=[0])
            x_var = x.var(dim=[0], unbiased=False)

            # self.eps is a small constant to avoid division by 0
            x_norm = (x - x_mean[None, :])/torch.sqrt(x_var[None, :] + self.eps)
            
            # update running estimates of x_mean and unbiased x_var
            n = x.shape[0]
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * x_var * n / (n - 1)
            
        else:
            '''Your codes here'''
            # self.eps is a small constant to avoid division by 0
            x_norm = (x - self.running_mean[None, :])/torch.sqrt(self.running_var[None, :] + self.eps)
        
        # multiply weight/gamma and x_norm element-wise, then add bias/beta
        x_out = self.weight[None, :] * x_norm + self.bias[None, :]
            
        return x_out


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            '''Your codes here'''
            # x: shape = N x num_features x h x w
            x_mean = x.mean(dim=[0,2,3])
            x_var = x.var(dim=[0,2,3], unbiased=False)

            # self.eps is a small constant to avoid division by 0
            x_norm = (x - x_mean[None, :, None, None])/torch.sqrt(x_var[None, :, None, None] + self.eps)
            
            # update running estimates of x_mean and unbiased x_var
            n = x.shape[0] * x.shape[2] * x.shape[3]
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * x_var * n / (n - 1)
            
        else:
            '''Your codes here'''
            # self.eps is a small constant to avoid division by 0
            x_norm = (x - self.running_mean[None, :, None, None])/torch.sqrt(self.running_var[None, :, None, None] + self.eps)
        
        # multiply weight/gamma and x_norm element-wise, then add bias/beta
        x_out = self.weight[None, :, None, None] * x_norm + self.bias[None, :, None, None]
            
        return x_out


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
