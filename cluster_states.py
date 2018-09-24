# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:38:30 2017

@author: Camilo
"""
import numpy as np
from utils import clusterize_and_return_centers

X = np.load('states_as_nparray.npy')
num_states = 750
X_clustered = clusterize_and_return_centers(X, k=num_states, batch_size=100)
np.save('cluster_centers_'+str(num_states)+'.npy',X_clustered)