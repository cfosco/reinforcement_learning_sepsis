# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 04:58:50 2017

@author: Camilo
"""

import torch as t
from torch.nn import SELU, ReLU, BatchNorm1d as BN, Softplus, Sigmoid

class IntegralPredNet(t.nn.Module):
    
    def __init__(self, n_layers, hidden_dim, activation=SELU()):
        super(IntegralPredNet, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        
    
    