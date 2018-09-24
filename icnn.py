"""
This file defines our implementations of the ICNN using PyTorch, as well as functions useful for training, such as:
    - get_q_target, that allows to get the target r+ gamma*Q_e(s', argmax_a' Q_s(s', a'))
    - clipping gradient
    - some auxiliary losses
    - compute the integral of Q (to convert it to a policy)
You should rather use ICNNBN2 which is the most up to date class (and is supposed to be the same as in the TF code)

"""

import torch as t
from torch.nn import SELU, ReLU, BatchNorm1d as BN, Softplus, Sigmoid, LeakyReLU
from utils import variable
import numpy as np
from itertools import product
from copy import deepcopy
import math

sigmoid = Sigmoid()
softplus = Softplus()
relu = ReLU()


class ICNN(t.nn.Module):
    """
    CONCAVE Q network

    THE ACTION DIM IS HARDCODED TO 2 HERE
    """

    def __init__(self, n_layers, hidden_dim, input_dim, activation=SELU()):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        for l in range(n_layers - 1):
            if l > 0:
                setattr(self, 'u' + str(l), t.nn.Linear(hidden_dim, hidden_dim))
            else:
                setattr(self, 'u' + str(l), t.nn.Linear(input_dim, hidden_dim))
        for l in range(n_layers):
            output_dim = (l < n_layers - 1) * hidden_dim + (l == n_layers - 1) * 1
            idim = (l > 0) * hidden_dim + (l == 0) * input_dim
            setattr(self, 'z_u' + str(l), t.nn.Linear(idim, output_dim))
            setattr(self, 'z_au' + str(l), t.nn.Linear(idim, 2))
            setattr(self, 'z_au_' + str(l), t.nn.Linear(2, output_dim, bias=False))
            if l > 0:
                setattr(self, 'z_zu' + str(l), t.nn.Linear(idim, hidden_dim))
                setattr(self, 'z_zu_' + str(l), t.nn.Linear(hidden_dim, output_dim, bias=False))

        # initialize parameters correctly (see paper SELU)
        self.initialize_weights_selu()

        # enforce convexity (or rather concavity)
        self.make_cvx()

    def forward(self, s, a):
        u = s
        for l in range(self.n_layers):
            if l == 0:
                fc_u = getattr(self, 'z_u' + str(l))
                fc_au_ = getattr(self, 'z_au_' + str(l))
                fc_au = getattr(self, 'z_au' + str(l))
                z = self.activation(fc_u(u) + fc_au_(fc_au(u) * a))
            else:
                fc_u = getattr(self, 'z_u' + str(l))
                fc_au_ = getattr(self, 'z_au_' + str(l))
                fc_au = getattr(self, 'z_au' + str(l))
                fc_zu_ = getattr(self, 'z_zu_' + str(l))
                fc_zu = getattr(self, 'z_zu' + str(l))
                z = fc_u(u) + fc_au_(fc_au(u) * a) + fc_zu_(ReLU()(fc_zu(u)) * z)
                if l < self.n_layers - 1:
                    z = self.activation(z)
            if l < self.n_layers - 1:
                fc = getattr(self, 'u' + str(l))
                u = self.activation(fc(u))
        return -z

    def initialize_weights_selu(self):
        for i in range(0, self.n_layers - 1):
            fcc = getattr(self, 'u' + str(i))
            shape = fcc.weight.data.numpy().shape
            fcc.weight.data = t.from_numpy(np.random.normal(0, 1 / np.sqrt(shape[0]), shape)).float()
        for i in range(self.n_layers):
            fcc = getattr(self, 'z_u' + str(i))
            shape = fcc.weight.data.numpy().shape
            fcc.weight.data = t.from_numpy(np.random.normal(0, 1 / np.sqrt(shape[0]), shape)).float()

            fcc = getattr(self, 'z_au' + str(i))
            shape = fcc.weight.data.numpy().shape
            fcc.weight.data = t.from_numpy(np.random.normal(0, 1 / np.sqrt(shape[0]), shape)).float()

            fcc = getattr(self, 'z_au_' + str(i))
            shape = fcc.weight.data.numpy().shape
            fcc.weight.data = t.from_numpy(np.random.normal(0, 1 / np.sqrt(shape[0]), shape)).float()

            if i > 0:
                fcc = getattr(self, 'z_zu' + str(i))
                shape = fcc.weight.data.numpy().shape
                fcc.weight.data = t.from_numpy(np.random.normal(0, 1 / np.sqrt(shape[0]), shape)).float()

                fcc = getattr(self, 'z_zu_' + str(i))
                shape = fcc.weight.data.numpy().shape
                fcc.weight.data = t.from_numpy(np.random.normal(0, 1 / np.sqrt(shape[0]), shape)).float()

    def make_cvx(self):
        """Make the neural network convex by absoluvaluing its W_zu weights"""
        for l in range(1, self.n_layers):
            w = getattr(self, 'z_zu'+str(l)).weight.data
            w.abs_()

    def proj(self):
        """If some weights became positive, set them to 0"""
        for l in range(1, self.n_layers):
            w = getattr(self, 'z_zu'+str(l)).weight.data
            w += w.abs()
            w /= 2


class ICNNBN(t.nn.Module):
    """
    CONCAVE Q network with BATCH NORMALIZATION
    """

    def __init__(self, n_layers, hidden_dim, input_dim, action_dim=2, gain=math.sqrt(2 / 1.01), RMAX=15, activation=LeakyReLU(.01)):
        super().__init__()
        self.n_layers = n_layers
        self.RMAX = RMAX
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        for l in range(n_layers - 1):
            if l > 0:
                setattr(self, 'u' + str(l), t.nn.Linear(hidden_dim, hidden_dim))
            else:
                setattr(self, 'u' + str(l), t.nn.Linear(input_dim, hidden_dim))
            if l < n_layers - 2:
                setattr(self, 'bn_u' + str(l), BN(hidden_dim, momentum=.9, eps=1e-3))

        for l in range(n_layers):
            output_dim = (l < n_layers - 1) * hidden_dim + (l == n_layers - 1) * 1
            idim = (l > 0) * hidden_dim + (l == 0) * input_dim
            setattr(self, 'z_u' + str(l), t.nn.Linear(idim, output_dim))
            setattr(self, 'bn_z_u' + str(l), BN(num_features=output_dim, momentum=.9, eps=1e-3))
            setattr(self, 'z_au' + str(l), t.nn.Linear(idim, action_dim))
            setattr(self, 'bn_z_au' + str(l), BN(num_features=action_dim, momentum=.9, eps=1e-3))
            setattr(self, 'z_au_' + str(l), t.nn.Linear(action_dim, output_dim, bias=False))
            setattr(self, 'bn_z_au_' + str(l), BN(num_features=output_dim, momentum=.9, eps=1e-3))
            if l > 0:
                setattr(self, 'z_zu' + str(l), t.nn.Linear(idim, hidden_dim))
                setattr(self, 'bn_z_zu' + str(l), BN(num_features=hidden_dim, momentum=.9, eps=1e-3))
                setattr(self, 'z_zu_' + str(l), t.nn.Linear(hidden_dim, output_dim, bias=False))
                setattr(self, 'bn_z_zu_' + str(l), BN(num_features=output_dim, momentum=.9, eps=1e-3))

        # self.xavier_init(gain)
        self.initialize_weights_xavier(gain)

        # enforce convexity (or rather concavity)
        self.make_cvx()

    def forward(self, s, a):
        # RMAX = self.RMAX
        u = s*1
        for l in range(self.n_layers):
            # z_i+1 from z_i, u_i and a
            if l == 0:
                fc_u = getattr(self, 'z_u' + str(l))
                fc_au_ = getattr(self, 'z_au_' + str(l))
                fc_au = getattr(self, 'z_au' + str(l))
                bn_u = getattr(self, 'bn_z_u' + str(l))
                bn_au_ = getattr(self, 'bn_z_au_' + str(l))
                bn_au = getattr(self, 'bn_z_au' + str(l))
                z = self.activation(bn_u(fc_u(u)) + bn_au_(fc_au_(bn_au(fc_au(u)) * a)))
            else:
                fc_u = getattr(self, 'z_u' + str(l))
                fc_au_ = getattr(self, 'z_au_' + str(l))
                fc_au = getattr(self, 'z_au' + str(l))
                fc_zu_ = getattr(self, 'z_zu_' + str(l))
                fc_zu = getattr(self, 'z_zu' + str(l))
                bn_u = getattr(self, 'bn_z_u' + str(l))
                bn_au_ = getattr(self, 'bn_z_au_' + str(l))
                bn_au = getattr(self, 'bn_z_au' + str(l))
                bn_zu_ = getattr(self, 'bn_z_zu_' + str(l))
                bn_zu = getattr(self, 'bn_z_zu' + str(l))

                z = bn_u(fc_u(u)) + bn_au_(fc_au_(bn_au(fc_au(u)) * a)) + bn_zu_(fc_zu_(ReLU()(bn_zu(fc_zu(u))) * z))
                if l < self.n_layers - 1:
                    z = self.activation(z)

            # u_i+1 from u_i
            if l < self.n_layers - 1:
                fc = getattr(self, 'u' + str(l))
                u = fc(u)
                if l < self.n_layers - 2:
                    bn = getattr(self, 'bn_u' + str(l))
                    u = bn(ReLU()(u))
        return -z
        # return RMAX * (2 * sigmoid(-z) - 1)

    def initialize_weights_xavier(self, gain):
        for i in range(0, self.n_layers - 1):
            fcc = getattr(self, 'u' + str(i))
            t.nn.init.xavier_uniform(fcc.weight, gain=gain)
        for i in range(self.n_layers):
            fcc = getattr(self, 'z_u' + str(i))
            t.nn.init.xavier_uniform(fcc.weight, gain=gain)

            fcc = getattr(self, 'z_au' + str(i))
            t.nn.init.xavier_uniform(fcc.weight, gain=gain)

            fcc = getattr(self, 'z_au_' + str(i))
            t.nn.init.xavier_uniform(fcc.weight, gain=gain)

            if i > 0:
                fcc = getattr(self, 'z_zu' + str(i))
                t.nn.init.xavier_uniform(fcc.weight, gain=gain)

                fcc = getattr(self, 'z_zu_' + str(i))
                t.nn.init.xavier_uniform(fcc.weight, gain=gain)

    def make_cvx(self):
        """Make the neural network convex by absoluvaluing its W_zu weights"""
        for l in range(1, self.n_layers):
            w = getattr(self, 'z_zu'+str(l)).weight.data
            w.abs_()

    def proj(self):
        """If some weights became positive, set them to 0"""
        for l in range(1, self.n_layers):
            w = getattr(self, 'z_zu'+str(l)).weight.data
            w += w.abs()
            w /= 2


class ICNNBN2(t.nn.Module):
    """
    CONCAVE Q network with BATCH NORMALIZATION only on the u layers (as in the TF code)
    This one is supposed to be exactly like the TF code (I hope so)
    """

    def __init__(self, n_layers, hidden_dim, input_dim, action_dim=2, gain=math.sqrt(2 / 1.01), activation=LeakyReLU(.01)):
        super().__init__()
        self.n_layers = n_layers
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        for l in range(n_layers - 1):  # in the last block there is no u
            if l > 0:
                setattr(self, 'u' + str(l), t.nn.Linear(hidden_dim, hidden_dim))
            else:
                setattr(self, 'u' + str(l), t.nn.Linear(input_dim, hidden_dim))
            if l < n_layers - 2:  # the last u does not have activation nor BN
                setattr(self, 'bn_u' + str(l), BN(hidden_dim, momentum=.9, eps=1e-3))

        for l in range(n_layers):
            output_dim = (l < n_layers - 1) * hidden_dim + (l == n_layers - 1) * 1  # the last z has dim 1 (Q value)
            idim = (l > 0) * hidden_dim + (l == 0) * input_dim  # the first block has input dim input dim, the next ones hidden dim
            setattr(self, 'z_u' + str(l), t.nn.Linear(idim, output_dim))
            setattr(self, 'z_au' + str(l), t.nn.Linear(idim, action_dim))
            setattr(self, 'z_au_' + str(l), t.nn.Linear(action_dim, output_dim, bias=False))
            if l > 0:  # at the first block, z is not yet defined
                setattr(self, 'z_zu' + str(l), t.nn.Linear(idim, hidden_dim))
                setattr(self, 'z_zu_' + str(l), t.nn.Linear(hidden_dim, output_dim, bias=False))

        self.initialize_weights_xavier(gain)

        # enforce convexity (or rather concavity)
        self.make_cvx()

    def forward(self, s, a):
        u = s*1
        for l in range(self.n_layers):

            # z_i+1 from z_i, u_i and a
            if l == 0:
                fc_u = getattr(self, 'z_u' + str(l))
                fc_au_ = getattr(self, 'z_au_' + str(l))
                fc_au = getattr(self, 'z_au' + str(l))
                z = self.activation(fc_u(u) + fc_au_(fc_au(u) * a))
            else:
                fc_u = getattr(self, 'z_u' + str(l))
                fc_au_ = getattr(self, 'z_au_' + str(l))
                fc_au = getattr(self, 'z_au' + str(l))
                fc_zu_ = getattr(self, 'z_zu_' + str(l))
                fc_zu = getattr(self, 'z_zu' + str(l))

                z = fc_u(u) + fc_au_(fc_au(u) * a) + fc_zu_(ReLU()(fc_zu(u)) * z)
                if l < self.n_layers - 1:
                    z = self.activation(z)

            # u_i+1 from u_i
            if l < self.n_layers - 1:
                fc = getattr(self, 'u' + str(l))
                u = fc(u)
                if l < self.n_layers - 2:
                    bn = getattr(self, 'bn_u' + str(l))
                    u = bn(ReLU()(u))  # they use ReLU in the TF code
        return -z

    def initialize_weights_xavier(self, gain):
        for i in range(0, self.n_layers - 1):
            fcc = getattr(self, 'u' + str(i))
            t.nn.init.xavier_uniform(fcc.weight, gain=gain)
        for i in range(self.n_layers):
            fcc = getattr(self, 'z_u' + str(i))
            t.nn.init.xavier_uniform(fcc.weight, gain=gain)

            fcc = getattr(self, 'z_au' + str(i))
            t.nn.init.xavier_uniform(fcc.weight, gain=gain)

            fcc = getattr(self, 'z_au_' + str(i))
            t.nn.init.xavier_uniform(fcc.weight, gain=gain)

            if i > 0:
                fcc = getattr(self, 'z_zu' + str(i))
                t.nn.init.xavier_uniform(fcc.weight, gain=gain)

                fcc = getattr(self, 'z_zu_' + str(i))
                t.nn.init.xavier_uniform(fcc.weight, gain=gain)

    def make_cvx(self):
        """Make the neural network convex by absoluvaluing its W_zu_ weights"""
        for l in range(1, self.n_layers):
            w = getattr(self, 'z_zu_'+str(l)).weight.data
            w.abs_()

    def proj(self):
        """If some W_zu_ weights became positive, set them to 0"""
        for l in range(1, self.n_layers):
            w = getattr(self, 'z_zu_'+str(l)).weight.data
            w += w.abs()
            w /= 2


def get_q_target(Q_select, Q_eval, next_states, rewards, batch_size, done, min0=-1, max0=1, min1=-1, max1=1, gamma=1., max_steps_a=10, return_max_actions=True, implementation=0):
    """
    This function computes the targets.
    It is either r or r + gamma Q_e(s', argmax_a' Q_s(s',a'))
    There are two implementations of this function
    :param Q_select:
    :param Q_eval:
    :param next_states:
    :param rewards:
    :param batch_size:
    :param done: an array of booleans indicating whether the state is terminal or nt
    :param min0: minimum value of the first action
    :param max0: maximum value of the first action
    :param min1: minimum value of the 2nd action
    :param max1: maximum value of the 2nd action
    :param gamma:
    :param max_steps_a: the maximum number of optimizing steps
    :param return_max_actions: whether to return the actions that maximized Q_select or not. If true, it returns a tuple
    :param implementation: 0 or 1. 0 is better. It allows for different dimensionality of the actions
    :return:
    """
    if implementation == 0:
        Q_target = rewards.squeeze().data.numpy() * 1
        action_dim = Q_select.action_dim
        Q_select.eval()
        Q_eval.eval()

        # identify non-terminal states
        # mask = t.ByteTensor(tuple((1 - done).tolist())).squeeze()
        mask = t.from_numpy((1 - done)).byte().view(-1, 1)
        shape = int(next_states.data.numpy().shape[1])
        masked_next_states = next_states.masked_select(mask).resize(int((1 - done).sum()), shape)

        # maximize
        max_action = variable(np.zeros((len(masked_next_states), action_dim)), requires_grad=True).float()
        prev_action = variable(np.zeros((len(masked_next_states), action_dim)), requires_grad=True).float()
        input_param = t.nn.Parameter(max_action.data)
        optimizer_for_a = t.optim.Rprop([input_param], lr=5e-1)
        for k in range(max_steps_a):
            max_action, input_param, optimizer_for_a = gradient_step_action(Q_select, masked_next_states, max_action, min0, max0, min1, max1, input_param=input_param, c=1e6, optimizer=optimizer_for_a)
            if np.max(np.abs(prev_action.data.numpy() - max_action.data.numpy())) < 1e-3:
                break
            prev_action = max_action * 1

        pred = Q_eval.forward(masked_next_states, max_action).squeeze()

        np_mask = mask.squeeze().numpy()
        np_mask = [k for k in range(batch_size) if np_mask[k] == 1]
        Q_target[np_mask] += gamma * (pred.data.numpy())

        Q_select.train()

        if return_max_actions:
            return variable(Q_target), max_action.data.numpy()
        else:
            return variable(Q_target)
    elif implementation == 1:
        action_dim = Q_select.action_dim
        next_state_ = []
        good = dict()
        for i, s in enumerate(next_states):
            if not np.isnan(s.data.numpy()[0]):
                next_state_.append(s.resize(1, 50))
                good[i] = len(good)
        if len(good) == 0:
            return rewards.squeeze(), variable(np.array(len(rewards) * [[np.nan, np.nan]]))
        else:
            next_state_ = t.cat(next_state_)
            max_action = variable(np.zeros((len(next_state_), action_dim)), requires_grad=True).float()
            prev_action = variable(np.zeros((len(next_state_), action_dim)), requires_grad=True).float()
            input_param = t.nn.Parameter(max_action.data)
            optimizer_for_a = t.optim.Rprop([input_param], lr=5e-1)
            for k in range(max_steps_a):
                max_action, input_param, optimizer_for_a = gradient_step_action(Q_select, next_state_, max_action, min0, max0, min1, max1, input_param=input_param, optimizer=optimizer_for_a)
                if np.max(np.abs(prev_action.data.numpy() - max_action.data.numpy())) < 1e-3:
                    break
                prev_action = max_action * 1

            pred = Q_eval.forward(next_state_, max_action)
            Qvalues = [pred[good[i]].float().resize(1, 1) if i in good else variable(np.zeros((1, 1))).float() for i in range(len(next_state))]
            max_prev_Q_value = t.cat(Qvalues, dim=0)
            Q_target = rewards.squeeze() + gamma * max_prev_Q_value.squeeze()
            if not return_max_actions:
                return Q_target
            else:
                return Q_target, max_action
    else:
        raise ValueError('There are only two implementations of this function: 0 and 1')


def clip_gradients(icnn, bound=10):
    for p in icnn.parameters():
        if p.grad is not None:
            p.grad = p.grad * ((bound <= p.grad).float()) * ((bound >= p.grad).float()) + bound * ((p.grad > bound).float()) - bound * ((p.grad < -bound).float())


def argmin(icnn, s, min0, max0, min1, max1):
    """
    The minimum of the icnn on the rectangle is attained in one of the corners because it is concave
    """
    return float(min([
        icnn.forward(s, variable([min0, min1])).data.numpy()[0],
        icnn.forward(s, variable([min0, max1])).data.numpy()[0],
        icnn.forward(s, variable([max0, min1])).data.numpy()[0],
        icnn.forward(s, variable([max0, max1])).data.numpy()[0]
    ]))


def compute_integral_general(func, s, min0, max0, min1, max1, min_value=None, nsteps=20):
    if min_value == None:
        #        print('type func', type(func), 'type s:', type(s), 'min0 max0', min0,max0)
        min_value = argmin(func, s, min0, max0, min1, max1)
    #    print('min_value of icnn:', min_value)
    grid_0 = np.linspace(min0, max0, nsteps)  # go from min_vaso to max_vaso in nsteps
    grid_1 = np.linspace(min1, max1, nsteps)  # go from min_iv to max_iv in nsteps

    action_grid = variable(np.array([list(x) for x in product(grid_0, grid_1)]))  # put a grid upon the action space. The icnn values are computed at each point of the grid
    copy_state = t.cat(nsteps ** 2 * [s.resize(1, 50)])  # duplicate the state nsteps^2 times (one per point in the grid)
    values = func(copy_state, action_grid)

    integral = t.sum(values) * (max1 - min1) * (max0 - min0) / nsteps ** 2  # approximation of the integral on the square (Riemann sum)
    #    print('integral value:', integral)
    return integral


def compute_integral(icnn, s, min0, max0, min1, max1, min_value=None, nsteps=20):
    """Compute the integral of icnn(s,a) - min_a icnn(s,a) for a given s"""
    if min_value == None:
        min_value = argmin(icnn.forward, s, min0, max0, min1, max1)
    #    print('min_value of icnn:', min_value)
    grid_0 = np.linspace(min0, max0, nsteps)  # go from min_vaso to max_vaso in nsteps
    grid_1 = np.linspace(min1, max1, nsteps)  # go from min_iv to max_iv in nsteps

    action_grid = variable(np.array([list(x) for x in product(grid_0, grid_1)]))  # put a grid upon the action space. The icnn values are computed at each point of the grid
    copy_state = t.cat(nsteps ** 2 * [s.resize(1, 50)])  # duplicate the state nsteps^2 times (one per point in the grid)
    values = icnn.forward(copy_state, action_grid)
    #    print('values in compute_integral: (look at type and dims)',values)
    #    print( '-- min value:', min_value, 'number of values < min_value:',sum(values.data.numpy()<min_value),'values<min_value:', values.data.numpy()[values.data.numpy()<min_value])
    integral = t.sum(values - min_value) * (max1 - min1) * (max0 - min0) / nsteps ** 2  # approximation of the integral on the square (Riemann sum)
    #    print('integral value:', integral)
    return integral


def discretize_Q(icnn, s, Q_integral, min0, max0, min1, max1):
    """Compute a discrete value of Q(s,.) by discretizing the outputs of the icnn (the approximated Q function) divided by its integral,
        to transform the Q function into a distribution that agrees with the concept of policy (gives prob of each action for state s)"""
    nsteps = 5
    min_value = argmin(icnn, s, min0, max0, min1, max1)
    grid_0 = np.linspace(min0, max0, nsteps)  # go from min_vaso to max_vaso in nsteps
    grid_1 = np.linspace(min1, max1, nsteps)  # go from min_iv to max_iv in nsteps

    #    print('grid_0',grid_0)
    #    print('grid_1',grid_1)

    action_grid = variable(np.array([list(x) for x in product(grid_0, grid_1)]))  # put a grid upon the action space. The icnn values are computed at each point of the grid
    copy_state = t.cat(nsteps ** 2 * [s.resize(1, 50)])  # duplicate the state nsteps^2 times (one per point in the grid)

    #    print('copy_state:', copy_state)
    #    print('action_grid:', action_grid)
    values = icnn.forward(copy_state, action_grid)
    #    print('values from icnn.forward:', values)
    assert (len(values) == 25)
    #    if Q_integral.data.numpy() != 0:
    return (values - min_value) / Q_integral


#    else:
#        return (values-min_value)


def update_parameters_lag(Q_select, Q_eval, tau):
    """
    theta_eval^i+1 += tau * (theta_select^i+1 - theta_eval^i)
    """
    for l in range(Q_select.n_layers - 1):
        u_select = getattr(Q_select, 'u' + str(l))
        u_eval = getattr(Q_eval, 'u' + str(l))
        u_eval.weight.data = (1 - tau) * u_eval.weight.data + tau * u_select.weight.data

    for l in range(Q_select.n_layers):
        output_dim = (l < Q_select.n_layers - 1) * Q_select.hidden_dim + (l == Q_select.n_layers - 1) * 1
        zu_select = getattr(Q_select, 'z_u' + str(l))
        zu_eval = getattr(Q_eval, 'z_u' + str(l))
        zu_eval.weight.data = (1 - tau) * zu_eval.weight.data + tau * zu_select.weight.data

        zau_select = getattr(Q_select, 'z_au' + str(l))
        zau_eval = getattr(Q_eval, 'z_au' + str(l))
        zau_eval.weight.data = (1 - tau) * zau_eval.weight.data + tau * zau_select.weight.data

        zau__select = getattr(Q_select, 'z_au_' + str(l))
        zau__eval = getattr(Q_eval, 'z_au_' + str(l))
        zau__eval.weight.data = (1 - tau) * zau__eval.weight.data + tau * zau__select.weight.data

        if l > 0:
            zzu_select = getattr(Q_select, 'z_zu' + str(l))
            zzu_eval = getattr(Q_eval, 'z_zu' + str(l))
            zzu_eval.weight.data = (1 - tau) * zzu_eval.weight.data + tau * zzu_select.weight.data

            zzu__select = getattr(Q_select, 'z_zu_' + str(l))
            zzu__eval = getattr(Q_eval, 'z_zu_' + str(l))
            zzu__eval.weight.data = (1 - tau) * zzu__eval.weight.data + tau * zzu__select.weight.data


def freeze_weights(icnn, frozen):
    """
    Freeze (or unfreeze) the weights of the ICNN
    :param icnn:
    :param frozen: whether to freeze the weights or not
    :return:
    """
    for l in range(icnn.n_layers):
        if l < icnn.n_layers - 1:
            fc = getattr(icnn, 'u' + str(l))
            fc.weight.requires_grad = not frozen
        if l == 0:
            fc_u = getattr(icnn, 'z_u' + str(l))
            fc_au_ = getattr(icnn, 'z_au_' + str(l))
            fc_au = getattr(icnn, 'z_au' + str(l))
            fc_u.weight.requires_grad = not frozen
            fc_au_.weight.requires_grad = not frozen
            fc_au.weight.requires_grad = not frozen
        else:
            fc_u = getattr(icnn, 'z_u' + str(l))
            fc_au_ = getattr(icnn, 'z_au_' + str(l))
            fc_au = getattr(icnn, 'z_au' + str(l))
            fc_zu_ = getattr(icnn, 'z_zu_' + str(l))
            fc_zu = getattr(icnn, 'z_zu' + str(l))
            fc_u.weight.requires_grad = not frozen
            fc_au_.weight.requires_grad = not frozen
            fc_au.weight.requires_grad = not frozen
            fc_zu_.weight.requires_grad = not frozen
            fc_zu.weight.requires_grad = not frozen


def loss_range(a, min0, max0, min1, max1, c):
    """Penalty for not being in a given range"""
    if len(a) < 2:
        return c * (
            (a[:] > max0).float() * t.abs(a[:] - max0) +
            (a[:] < min0).float() * t.abs(a[:] - min0) +
            (a[:] > max1).float() * t.abs(a[:] - max1) +
            (a[:] < min1).float() * t.abs(a[:] - min1)
        )
    return c * (
        (a[:, 0] > max0).float() * t.abs(a[:, 0] - max0) +
        (a[:, 0] < min0).float() * t.abs(a[:, 0] - min0) +
        (a[:, 1] > max1).float() * t.abs(a[:, 1] - max1) +
        (a[:, 1] < min1).float() * t.abs(a[:, 1] - min1)
    )


def gradient_step_action(Q, s, a, min0, max0, min1, max1, c=1e4, input_param=None, optimizer=None):
    """
    Compute the gradients with respect to the action and update the action
    The first pass of this function defines the optimizer and the input param if it has not already been done

    :param Q: The Q network. WATCH OUT! IT SHOULD BE `-icnn` AND NOT `icnn`
    :param s: a torch Variable representing the state
    :param a: a torch Variable representing the actions. IT SHOULD HAVE `requires_grad=True`
    :param input_param: It is just the same thing as `a`, but wrapped into a pytorch Parameter (t.nn.Parameter(a))
    :param optimizer: the optimizer (by default use Adam)
    :returns the updated value of `a`, the updated value of `input_param` (a Parameter wrapping `a`), the optimizer

    Example use case:
    ```
        a = variable(np.zeros((1,2)), requires_grad=True).float()
        s = variable(np.zeros((1,50))).float()

        input_param = None
        optimizer = None
        for k in range(200):
            a, input_param, optimizer = gradient_step_action(icnn, s, a, input_param=input_param, optimizer=optimizer)
            print(a)
    ```
    """
    if input_param is None or optimizer is None:
        input_param = t.nn.Parameter(a.data)
        optimizer = t.optim.Rprop([input_param], lr=5e-1)

    assert len(s) == len(a), 'There should be as many states as there are actions'
    batch_size = len(s)

    # erase previous gradients
    optimizer.zero_grad()

    # trick to get the gradients wrt `a`
    grad = {}

    def f(x):
        grad['a'] = x

    a.register_hook(f)

    # get output (we want to maximize Q, so minimize -Q (the optimizer minimizes by default))
    output = -Q(s, a) + loss_range(a, min0, max0, min1, max1, c)

    # compute gradients
    output.backward(t.FloatTensor(batch_size * [[1.]]))

    # use the gradients that was deceitfully obtained using the hook
    input_param.grad = grad['a']

    # update the action
    optimizer.step()

    # returns the new value of `a` (a pytorch variable), the same thing but wrapped in t.nn.Parameter, and the optimizer
    return variable(input_param.data, requires_grad=True), input_param, optimizer


def diff_params(current_params, params0):
    """Check if the parameters of the network have been updated"""
    return t.sum(t.cat([t.sum((x1 - x2) ** 2).resize(1, 1) for x1, x2 in zip(current_params, params0)], 0)).data.numpy()[0]


def loss_beyond_RMAX(x, RMAX):
    """Penalty for not being in the range -RMAX, RMAX"""
    return (x > RMAX).float() * (x - RMAX) ** 2 + (x < -RMAX).float() * (x + RMAX) ** 2
