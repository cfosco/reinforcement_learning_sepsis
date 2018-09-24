
# coding: utf-8

# In[1]:

import numpy as np
from random import random, sample
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from IPython.display import display, clear_output
from parse_dataset import *
from copy import deepcopy
from time import time
import evaluation
import gym
import math
from utils import moving_avg
from agent import *
import tensorflow as tf
from icnn_tf import Agent


# In[2]:

env = gym.make('MountainCarContinuous-v0')
transitions = []
episodes = 1000
iterations = 100

for e in range(episodes):
    s = env.reset()
    for i in range(iterations):
        action = np.random.uniform(-1,1,(1,))
        former = env.env.state*1
        s_, reward, done, info = env.step(action)
        reward *= (reward>0)*.01  # to mke the pb simpler
        transitions.append((s, action, reward, s_, done))
        s = s_*1
        if done:
            break

env.reset()


# In[3]:

beta0 = .9
max_steps_beta = int(5e4)
beta = lambda t: (t<max_steps_beta)*(beta0 + t*(1-beta0)/max_steps_beta) + (t>=max_steps_beta)*1.
agent = Agent(2, 1, beta, [20, 20])

# In[5]:

for s,a,r,s_,done in transitions:
    agent.rm.add(*(s,a,r,s_,np.array([done])))


# In[6]:

# monitoring
losses = []
td_errors = []
test_td_errors = []
test_average_q_pred = []
test_max_q_pred = []
test_min_q_pred = []
average_q_pred = []
average_q_target = []
max_q_pred = []
min_q_pred = []
max_q_target = []
min_q_target = []
average_a = []
min_a = []
max_a = []


# In[7]:

plt.rcParams['figure.figsize'] = (9,9)
global_step = -1
max_steps_beta = 100000
beta0 = .9
beta = lambda t: (t<max_steps_beta)*(beta0 + t*(1-beta0)/max_steps_beta) + (t>=max_steps_beta)*1.
all_rewards = [0]

ss = env.reset()

while True:
    global_step += 1
    
    # act
    a = agent.act()
    ss_, reward, done, info = env.step(a)
    reward *= (reward > 0)*.01
    all_rewards.append(all_rewards[-1]+reward)
    agent.rm.add(*(ss, a, reward, ss_, np.array([done])))
    ss = ss_*1
    if done:
        ss = env.reset()
    if agent.t > 10000:
        env.render()

    loss, td_error, q_entr, q_target = agent.train()
    
    # Monitoring
    losses.append(loss.mean())
    td_errors.append(td_error.mean())
    average_q_pred.append(q_entr.mean())
    max_q_pred.append(q_entr.max())
    min_q_pred.append(q_entr.min())
    average_q_target.append(q_target.mean())
    max_q_target.append(q_target.max())
    min_q_target.append(q_target.min())
    average_a.append(a[0])

    if len(average_a) > 5000 and (.75 >= np.mean(average_a[-5000:]) >= -.75):
        try:
            if last_update <= global_step - 2000:
                last_update = global_step*1
                agent.save('tensorboard/models/' + str(last_update))
        except:
            last_update = global_step*1
            agent.save('tensorboard/models/' + str(last_update))
