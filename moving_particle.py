from random import random
import torch as t
from utils import variable
import numpy as np
from icnn import gradient_step_action

"""
A simple 2d RL problem
The agent is a particle moving in the square [-1,1]^2
It bounces on walls
The actions are the 2D acceleration
"""


class Agent:
    """
    The agent can move in the square [-1,1]^2
    Its speed is between -1 and 1
    Its acceleration (the actions) is also between -1 and 1
    It gets rewards +1 if it arrives in a zone defined by reward. IT ENDS THE EPISODE

    It bounces on the walls when it hits them. In that case, the speed is multiplied by -1 and the particle
    arrives at the same distance of the wall as if it had went through it

    The states consist in (x,y,vx,vy)
    The actions consist in (ax,ay)
    """
    dt = .1
    xlim = 1
    ylim = 1

    def __init__(self, x, y, vx, vy, reward, done):
        xlim = self.xlim
        ylim = self.ylim
        self.x = x
        self.y = y
        assert -xlim <= x <= xlim
        assert -ylim <= y <= ylim
        assert -1 <= vx <= 1
        assert -1 <= vy <= 1
        self.vx = vx
        self.vy = vy
        self.ax = 0
        self.ay = 0
        self.reward = reward  # a function of x,y,ax,ay
        self.done = done  # function of x,y

    def move(self, ax, ay):
        # pour ne pas avoir a ecrire self tout le temps
        dt = self.dt
        xlim = self.xlim
        ylim = self.ylim

        # rebondit sur les murs
        # update la vitesse. Si rebondissement, change le signe de la vitesse
        new_x = (self.x + self.vx * dt) * 1
        if -xlim <= new_x <= xlim:
            self.x += self.vx * dt
            self.vx += self.ax * dt
        elif new_x < -xlim:
            self.x = -xlim - (new_x + xlim)
            self.vx = -(self.vx + self.ax * dt)
        else:
            self.x = xlim - ((self.x + self.vx * dt) - xlim)
            self.vx = -(self.vx + self.ax * dt)

        new_y = (self.y + self.vy * dt) * 1
        if -ylim <= new_y <= ylim:
            self.y += self.vy * dt
            self.vy += self.ay * dt
        elif new_y < -ylim:
            self.y = -ylim - (new_y + ylim)
            self.vy = -(self.vy + self.ay * dt)
        else:
            self.y = ylim - ((self.y + self.vy * dt) - ylim)
            self.vy = -(self.vy + self.ay * dt)
        assert -xlim <= self.x <= self.xlim, (self.x, new_x, -xlim - (new_x + xlim))
        assert -ylim <= self.y <= self.ylim, (self.y, new_y, -ylim - (new_y + ylim))

        # clip speed if it goes beyond -1,1
        self.vx = 1. if self.vx > 1 else self.vx
        self.vx = -1. if self.vx < -1 else self.vx
        self.vy = 1. if self.vy > 1 else self.vy
        self.vy = -1. if self.vy < -1 else self.vy

        # update acceleration
        self.ax = ax
        self.ay = ay

        # get reward if inside the smaller square
        return self.reward(self.x, self.y, ax, ay), self.x, self.y, self.vx, self.vy, self.done(self.x, self.y)

    def observe(self):
        return self.x, self.y, self.vx, self.vy


def create_dataset(n_episodes, n_steps, reward, done):
    """
    Create a list of transition (s,a,r,s')
    It contains `n_episodes` of length `n_steps`
    """
    dataset = []
    for _ in range(n_episodes):
        x0, y0, vx0, vy0 = 2 * random() - 1, 2 * random() - 1, 2 * random() - 1, 2 * random() - 1
        agent = Agent(x0, y0, vx0, vy0, reward, done)
        is_done = done(x0, y0)

        for step in range(n_steps):
            if is_done:
                break
            # observe
            s = list(agent.observe())

            # choose action
            ax = 2 * random() - 1
            ay = 2 * random() - 1

            # move, reward, observe next state
            r, x, y, vx, vy, is_done = agent.move(ax, ay)

            # store transition
            dataset.append((_, s, (ax, ay), r, list((x, y, vx, vy)), is_done))
    return dataset


def argmax(Q_select, s, max_steps_a=10, min0=-1, max0=1, min1=-1, max1=1):
    """Choose action greedily wrt Q_select in state s"""
    Q_select.eval()
    # maximize
    max_action = variable(np.zeros((len(s), 2)), requires_grad=True).float()
    prev_action = variable(np.zeros((len(s), 2)), requires_grad=True).float()
    input_param = t.nn.Parameter(max_action.data)
    optimizer_for_a = t.optim.Rprop([input_param], lr=5e-1)
    for k in range(max_steps_a):
        max_action, input_param, optimizer_for_a = gradient_step_action(Q_select, s, max_action, min0, max0, min1, max1, input_param=input_param, optimizer=optimizer_for_a)
        if np.max(np.abs(prev_action.data.numpy() - max_action.data.numpy())) < 1e-3:
            break
        prev_action = max_action * 1
    return max_action


def clip(x, lower_bound, upper_bound):
    """
    Clip values below -RMAX or above RMAX
    Might be useful when the target goes beyond
    """
    l = lower_bound
    u = upper_bound
    return x*((x>=l).float())*((x<=u).float()) + u*(x>u).float() + l*(x<l).float()
