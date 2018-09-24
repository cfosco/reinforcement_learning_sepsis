# -*- coding: utf-8 -*-
"""
Evaluation functions for the Sepsis ICNN approach

@author: Camilo
"""
import numpy as np
import warnings
from utils import variable
#from icnn import argmin, variable, product
import importlib
import icnn as ICNN
from scipy import spatial
#from icnn import compute_integral, discretize_Q
importlib.reload(ICNN)
from timeit import default_timer as timer
from tqdm import tqdm
import torch
import parse_dataset as parse
importlib.reload(parse)




def eval_ICNN_WDR_continuous(icnn, extremums_of_action_space, cont_states_sequence, cont_actions_sequence, rewards_sequence, 
                             quantiles_fluid, quantiles_vaso, fence_posts, gamma, q_clinician = None):
    if q_clinician == None:
        q_clinician = np.load('q_clinician.npy')
#    if state_cluster_centers == None:
#        state_cluster_centers = np.load('cluster_centers_750.npy')
#    if action_bins == None:
#        action_bins = np.load('action_bins.npy')

    
    
    pi_behavior = build_pi_behavior_function_full_interp(q_clinician, quantiles_fluid, quantiles_vaso)#discrete_states, cont_states_sequence, discrete_actions, cont_actions_sequence)
    
    pi_evaluation = build_pi_evaluation_function(icnn, extremums_of_action_space)
    
    print('Starting WDR calculations')
    return continuous_weighted_doubly_robust(cont_states_sequence, cont_actions_sequence, rewards_sequence, fence_posts, 
                                             gamma, pi_evaluation, pi_behavior, icnn, extremums_of_action_space)
    
    
def eval_ICNN_WDR_general(Q_func, extremums_of_action_space, cont_states_sequence, cont_actions_sequence, rewards_sequence, 
                             quantiles_fluid, quantiles_vaso, fence_posts, gamma, q_clinician = None):
    if q_clinician == None:
        q_clinician = np.load('q_clinician.npy')

    pi_behavior = build_pi_behavior_function_full_interp(q_clinician, quantiles_fluid, quantiles_vaso)#discrete_states, cont_states_sequence, discrete_actions, cont_actions_sequence)
    pi_evaluation = build_pi_evaluation_function_general(Q_func, extremums_of_action_space)
    
    print('Starting WDR calculations')
    return continuous_weighted_doubly_robust(cont_states_sequence, cont_actions_sequence, rewards_sequence, fence_posts, 
                                             gamma, pi_evaluation, pi_behavior, Q_func, extremums_of_action_space)
    
       
def eval_ICNN_WDR_mountaincar(Q_func, pi_behavior, extremums_of_action_space, cont_states_sequence, cont_actions_sequence, 
                              rewards_sequence, fence_posts, gamma):
    
    pi_evaluation = build_pi_evaluation_function_mountaincar(Q_func, extremums_of_action_space)
    
    print('Starting WDR calculations')
    return continuous_weighted_doubly_robust(cont_states_sequence, cont_actions_sequence, rewards_sequence, fence_posts, 
                                             gamma, pi_evaluation, pi_behavior, Q_func, extremums_of_action_space, mountaincar=True)

def build_pi_evaluation_function(icnn, extremums_of_action_space):
    
    min0, max0, min1, max1 = extremums_of_action_space
    def pi_evaluation(s,a):
#        print('s:',s)
        icnn_min = ICNN.argmin(icnn, s, min0, max0, min1, max1)
        value = (icnn.forward(s,a)-icnn_min)/ICNN.compute_integral(icnn, s, min0, max0, min1, max1, min_value = icnn_min) 
        return value.data.numpy()  
    return pi_evaluation


def build_pi_evaluation_function_mountaincar(Q_func, extremums_of_action_space):
    
    def pi_evaluation(s,a):
        icnn_min = Q_min_1D(Q_func, s, extremums_of_action_space)
        value = (Q_func(s,a)-icnn_min)/compute_integral_1D(Q_func, s, extremums_of_action_space, min_value = icnn_min) 
        return value  
    return pi_evaluation

def build_pi_evaluation_function_general(Q_func, extremums_of_action_space):
    
    min0, max0, min1, max1 = extremums_of_action_space
    def pi_evaluation(s,a):
        
        Q_min = Q_minimize(Q_func, s, min0, max0, min1, max1)
        integ = ICNN.compute_integral_general(Q_func, s, min0, max0, min1, max1, min_value = Q_min) 
        print('integ',integ)
        value = (Q_func(s,a)-Q_minimize)/integ
        return value.data.numpy()  
    return pi_evaluation

def Q_minimize(Q_func, s, extremums_of_action_space):
    """
    The minimum of the icnn on the action space is attained on one of the edges of that space, because it is concave
    """
    
    min0, max0, min1, max1 = extremums_of_action_space
    
    return float(min([
        Q_func(s, variable([min0, min1])).data.numpy()[0],
        Q_func(s, variable([min0, max1])).data.numpy()[0],
        Q_func(s, variable([max0, min1])).data.numpy()[0],
        Q_func(s, variable([max0, max1])).data.numpy()[0]
    ]))

def compute_integral_1D(Q_func, s, extremums_of_action_space, min_value = [], n_steps = 20):
    """Compute the integral of icnn(s,a) - min_a icnn(s,a) for a given s"""
    
#    print('s in compute integral 1d',s)
    if s.shape == (2,):
        s=s.reshape(1,2)
        
    integral = []
    for state in s:
        if min_value == []:
            min_value = Q_min_1D(Q_func, state, extremums_of_action_space)
        min_a, max_a = extremums_of_action_space
        a_grid = np.linspace(min_a, max_a, n_steps).reshape((n_steps,1))
        values = Q_func(np.array([state]*n_steps), a_grid)
    #    print('values in compute_integral: (look at type and dims)',values)
        integral.append( sum(values - min_value) * (max_a-min_a) / n_steps ) # Integral approximation
    #    print('integral value:', integral)
    return integral      
    

def Q_min_1D(Q_func, s, extremums_a_space):
    """
    The minimum of the icnn is attained on one of the edges of the action space.
    """
    
    if s.shape == (2,):
        s=s.reshape(1,2)
    
    min_a_repeated = np.array([extremums_a_space[0]]*s.shape[0]).reshape(s.shape[0],1)
    max_a_repeated = np.array([extremums_a_space[1]]*s.shape[0]).reshape(s.shape[0],1)
    
#    print('s.shape',s.shape)
#    print('min_a_repeated.shape',min_a_repeated.shape)
    Q_output_min = Q_func(s, min_a_repeated)
    Q_output_max = Q_func(s, max_a_repeated)
        
#    print('Q_func(s,np.array([[extremums_a_space[0]]]))',Q_func(s,np.array([[extremums_a_space[0]]])))

    return np.min(np.concatenate((Q_output_min.reshape(s.shape[0],1), Q_output_max.reshape(s.shape[0],1)),axis=1),axis=1)
    

def build_pi_behavior_function(q_clinician, quantiles_fluid, quantiles_vaso, cluster_centers = None):   #discrete_states, cont_states, discrete_actions, cont_actions):
    
#    state_cont_to_disc_dictionary = {}
#    action_cont_to_disc_dictionary = {}
    if cluster_centers == None:
        cluster_centers = np.load('cluster_centers_750.npy')
    kd_tree = spatial.KDTree(cluster_centers)
#    for i in range(len(discrete_states)):
#        state_cont_to_disc_dictionary[cont_states[i]] = discrete_states[i]
#    for i in range(len(discrete_actions)):
#        action_cont_to_disc_dictionary[cont_actions[i]] = discrete_actions[i]
    
    num_states = q_clinician.shape[0]
    num_actions = q_clinician.shape[1]
    pi_behavior_table = np.zeros((num_states, num_actions))
    for s in range(q_clinician.shape[0]):
        pi_behavior_table[s,np.argmax(q_clinician[s,:])]=1
        
    def pi_behavior(s,a):
        a_discrete = parse.quantized_actions(a.data.numpy()[0], a.data.numpy()[1], quantiles_fluid, quantiles_vaso, apply_log=False)
        s_discrete = kd_tree.query(s.data.numpy())[1]
#        print('kd_tree.query(s)[1]',kd_tree.query(s.data.numpy())[1])
#        s_discrete = state_cont_to_disc_dictionary[s.data.numpy()]
#        a_discrete = action_cont_to_disc_dictionary[a.data.numpy()]
        return pi_behavior_table[s_discrete,a_discrete]+1e-4
        
    return pi_behavior

def build_pi_behavior_function_semi_interp(q_clinician, quantiles_fluid, quantiles_vaso, cluster_centers = None):
    if cluster_centers == None:
        cluster_centers = np.load('cluster_centers_750.npy')
    kd_tree = spatial.KDTree(cluster_centers)
    num_states = q_clinician.shape[0]
    num_actions = q_clinician.shape[1]
    
    gauss_interp = {}
    
    for s_ in range(num_states):
        gauss_interp[s_]=[]
        for a_ in range(num_actions):
            gauss_interp[s_].append(build_gaussian_func(mu=parse.get_cont_action(a_, quantiles_fluid, quantiles_vaso),sig=1,scale=q_clinician[s_,a_]))
    
    def pi_behavior(s,a):
        s_discrete = kd_tree.query(s.data.numpy())[1]
        value = 0
        for g in gauss_interp[s_discrete]:
            value += g(a.data.numpy())
        return value
    
    return pi_behavior

def build_pi_behavior_function_full_interp(q_clinician, quantiles_fluid, quantiles_vaso, cluster_centers = None):
    num_states = q_clinician.shape[0]
    num_actions = q_clinician.shape[1]
    gauss_interp = []
    if cluster_centers == None:
        cluster_centers = np.load('cluster_centers_750.npy')
    
    for s_ in range(num_states):
        for a_ in range(num_actions):
            cont_a = parse.get_cont_action(a_, quantiles_fluid, quantiles_vaso)
            mu_s_a = cluster_centers[s_].extend(cont_a)
            gauss_interp.append( build_gaussian_func(mu=mu_s_a,sig=np.eye(len(mu_s_a)),scale=q_clinician[s_,a_])  ) 
    
    def pi_behavior(s,a):
        value = 0
        for g in gauss_interp:
            value += g(s.data.numpy().extend(a.data.numpy()))
        return value
    
    return pi_behavior


def build_gaussian_func(mu=[0,0], sig = np.eye(2), scale = 1):
    """Builds N-dimensional gaussian function with predefined mean, sigma and scale."""
    
    def gaussian(x):
        return scale/np.sqrt(np.linalg.det(2*np.pi*sig))*np.exp(-1/2*np.dot(np.dot((x - mu).T,np.linalg.inv(sig)),(x - mu)))
    
    return gaussian


def continuous_weighted_doubly_robust(cont_states_sequence, cont_actions_sequence, rewards_sequence, fence_posts, 
                                      gamma, pi_evaluation_func, pi_behavior_func, Q, extremums_of_action_space, mountaincar=False):
    
    """
    Computes a continuous WDR. 
    """
    
    start = timer()
    if mountaincar == True:
        V = build_V_function_mountaincar(Q, pi_evaluation_func, extremums_of_action_space)
    else:
        V = build_V_function(Q, pi_evaluation_func, extremums_of_action_space)
    num_of_trials = len( fence_posts )
    individual_trial_estimators = []

    # Compute the weights
    rho_array, weights_normalization = compute_weights_WDR(pi_evaluation_func, pi_behavior_func, num_of_trials, 
                                                           fence_posts, cont_states_sequence, cont_actions_sequence)
    # This loop computes the actual estimator
    print('Starting calculation of individual estimators (per episode)')
    for trial_i in tqdm(range( num_of_trials )):
        current_trial_estimator = 0
        rho = 1
        w = 1 / num_of_trials
        discount = 1/gamma
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len(cont_states_sequence) - fence_posts[-1]
        t_within_trial = 0
#        print('fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial', fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial)
        for t in range(fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
#            print('len states seq:',len(actions_sequence))
#            print('t',t)
            previous_w = w
            rho *= pi_evaluation_func(cont_states_sequence[t], cont_actions_sequence[t])/ \
                    pi_behavior_func(cont_states_sequence[t], cont_actions_sequence[t])
                
            # WARNING: CLIPPING RHO   
            rho=np.clip(rho, 0,3)
            
            w = rho / weights_normalization[ t_within_trial ]
            discount *= gamma
#            print('states_sequence[ t ], actions_sequence[ t ]:',states_sequence[ t ], actions_sequence[ t ])
#            print('shape Q, shape V',Q.shape,V.shape)
#            print('cont_states_sequence[ t ]',cont_states_sequence[ t ])
#            print('V( cont_states_sequence[ t ] )',V( cont_states_sequence[ t ] ))
#            print('previous_w',previous_w)  

            current_trial_estimator += w * discount * rewards_sequence[ t ] - \
                discount * ( w * Q( cont_states_sequence[ t ], cont_actions_sequence[ t ] ) - \
                             previous_w * V( cont_states_sequence[ t ])[0] )
            t_within_trial += 1
        individual_trial_estimators += [ current_trial_estimator ]
        print('Estimator for episode',trial_i,':', current_trial_estimator)
    estimator = np.sum( individual_trial_estimators )

    end = timer()
    print('Time elapsed in WDR calculation:', end-start)
    
    return estimator, individual_trial_estimators

def build_V_function_mountaincar(Q_func, pi_eval, ex_a_space):
    pi_times_Q = pi_times_Q_builder_mountaincar(pi_eval, Q_func)
    def V(s):
        return compute_integral_1D(pi_times_Q, s, ex_a_space)
    return V
    
def pi_times_Q_builder_mountaincar(pi_evaluation_func, Q_func):
    def pi_times_Q(s,a):
        return pi_evaluation_func(s,a)*Q_func(s,a)
    return pi_times_Q


def compute_weights_WDR(pi_evaluation_func, pi_behavior_func, num_of_trials, fence_posts, cont_states_sequence, cont_actions_sequence):
    fence_posts_with_length_appended = fence_posts + [ len( cont_states_sequence ) ]
    single_patient_sequences_length = [ fence_posts_with_length_appended[i+1] - \
        fence_posts_with_length_appended[i] for i in range(len(fence_posts)) ]
    length_of_longest_patient_sequence = max( single_patient_sequences_length )
    rho_array = np.nan * np.zeros( ( num_of_trials, length_of_longest_patient_sequence ) )
    
    # This loop computes the weights
    for trial_i in range( num_of_trials ):
        rho = 1
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( cont_states_sequence ) - fence_posts[-1]
        t_within_trial = 0
        for t in range(fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):

            rho *= pi_evaluation_func(cont_states_sequence[t], cont_actions_sequence[t])/ \
                    pi_behavior_func(cont_states_sequence[t], cont_actions_sequence[t])
            
            # WARNING: CLIPPING RHO   
            rho=np.clip(rho, 0,3)
            
            rho_array[ trial_i, t_within_trial ] = rho
            t_within_trial += 1
        rho_array[ trial_i, t_within_trial: ] = rho
    weights_normalization = np.sum( rho_array, axis = 0 )
    
    return rho_array, weights_normalization
    

def build_V_function(Q_func, pi_evaluation_func, extremums_of_action_space):
    min0, max0, min1, max1 = extremums_of_action_space
    pi_times_Q = pi_times_Q_builder(pi_evaluation_func, Q_func)
    def V(s):
        value = ICNN.compute_integral_general(pi_times_Q, s, min0, max0, min1, max1)
        return value.data.numpy()
    
    return V   
    
def pi_times_Q_builder(pi_evaluation_func, Q_func, pytorch=True):
    
    def pi_times_Q(s,a):
        arr=[]
#        print('len(s):',len(s))
#        print('len(a):',len(a))
        for i in range(len(s)):
            arr.append(variable(pi_evaluation_func(s[i],a[i]))*Q_func(s[i],a[i]))
        return torch.cat(arr)
    return pi_times_Q
    



















































def eval_ICNN_WDR_HW3(icnn, extremums_of_action_space, D, gamma, rmin, rmax):
    # We're calculating DR on a discretized case, so our num_states and num_actions should be discrete:
    q_clinician = np.load('q_clinician.npy')
    num_states = q_clinician.shape[0]
    num_actions = q_clinician.shape[1]
    pi_evaluation = np.zeros((num_states, num_actions))
#    pi_behavior = np.zeros((num_states, num_actions))
    state_values = np.load('cluster_centers_750.npy')
    Q_integral = 0
    for i,s in enumerate(state_values):
        min0, max0, min1, max1 = extremums_of_action_space
#        print('computing integral for state:',s)
#        Q_integral = compute_integral(icnn, variable(s), min0, max0, min1, max1)
#        print('type Q_integral', type(Q_integral), ' ---- Value Q_integral:',Q_integral)
#        pi_behavior[i,np.argmax(q_clinician[i,:])]=1
        # We want pi_evaluation to be a discretized policy. It will take a state as input
        # and will return a discrete proba distro over the 25 possible actions.
        ''' IMPORTANT TODO: discretize actions following the bins used in HW3 '''
        pi_evaluation[i,:] = ICNN.discretize_Q(icnn, variable(s), Q_integral,  min0, max0, min1, max1).squeeze().data.numpy()
    
    # D is a list of episodes. Episodes have to be tuples of 4 floats.
    DR = eval_doubly_robust(num_states, num_actions, D, q_clinician, pi_evaluation, gamma, rmin, rmax)
    
    return DR        

def eval_ICNN_WDR_Omer(icnn, extremums_of_action_space, states_sequence, actions_sequence, 
                      rewards_sequence, fence_posts, trans_as_tuples, gamma, state_values=None):
    if state_values == None:
        state_values = np.load('cluster_centers_750.npy')
    
    q_clinician = np.load('q_clinician.npy')
    num_states = q_clinician.shape[0]
    num_actions = q_clinician.shape[1]
    pi_behavior_table = np.zeros((num_states,num_actions))
    pi_evaluation_table = np.zeros((num_states,num_actions))    
    min0, max0, min1, max1 = extremums_of_action_space
    for i,s in enumerate(state_values):
        
#        print('computing integral for state:',s)
        Q_integral = ICNN.compute_integral(icnn, variable(s), min0, max0, min1, max1)
#        print('Q_integral for state', i, ':',Q_integral.data.numpy())
#        print('type Q_integral', type(Q_integral), ' ---- Value Q_integral:',Q_integral)
        pi_behavior_table[i,np.argmax(q_clinician[i,:])]=1
        # We want pi_evaluation to be a discretized policy. It will take a state as input
        # and will return a discrete proba distro over the 25 possible actions.
        ''' IMPORTANT TODO: discretize actions following the bins used in HW3 '''
        pi_evaluation_table[i,:] = ICNN.discretize_Q(icnn, variable(s), Q_integral,  min0, max0, min1, max1).squeeze().data.numpy()
        
    print('pi_eval computed. First row:',pi_evaluation_table[0,:])
    print('How close is each row to an actual distribution? Lets see if the first 20 rows sum to one:', [sum(pi_evaluation_table[i,:] for i in range(20))])
    print('Now starting WDR calculations')
    
    return off_policy_per_decision_weighted_doubly_robust(states_sequence, actions_sequence, rewards_sequence, fence_posts, trans_as_tuples, gamma,
                                                   pi_evaluation_table, pi_behavior_table)



def eval_doubly_robust(num_states, num_actions, D, pi_behavior, pi_evaluation, gamma, rmin, rmax, max_t=200):
    DR = []
    build_approximate_model = make_approximate_model_builder(num_states, num_actions)
    
    # hyperparams
    R_MIN = rmin
    R_MAX = rmax
    RHO_MIN = 0
    RHO_MAX = 3
    C = np.zeros((max_t, len(D)))

    MAX_HORIZON = max_t
#    print('Starting DR calculations')
    for epi_i in tqdm(range(len(D))):
        episode = D[epi_i]
#        print('Episode:',episode)
        # using D until epi_i, build an approximate model
        T_hat, R_hat = build_approximate_model(episode)
#        print('T_hat.shape:',T_hat.shape)
#        print('R_hat.shape:',R_hat.shape)
#        print('pi_evaluation',pi_evaluation.shape)
        # evalaute pi_e on our imperfect model
        Q_hat_pi_e, V_hat_pi_e = solve_MDP_equations(T_hat, R_hat, gamma, pi_evaluation)
#        V_hat_pi_e = policy_evaluation( T_hat, R_hat, pi_evaluation, gamma)
#        Q_hat_pi_e = pi_evaluation.Q
        # compute importance weights
        rhos = []
        X = 0.0
        Y = 0.0
        Z = 0.0
        DR_i = 0.0
        #import pdb;pdb.set_trace()
        for t, exp in enumerate(episode[:MAX_HORIZON]):
            s, a, r, _ = exp
            # compute X: rewards scaled by importance weights
#            p1 = pi_evaluation.query_Q(s,a)
#            p2 = pi_behavior.query_Q(s,a)
            
            p1 = pi_evaluation[s,a]
            p2 = pi_behavior[s,a]

            rho = np.clip(p1/p2, RHO_MIN, RHO_MAX)

            rhos.append(rho)
            rho_t = np.prod(rhos[:t+1])
            # for WDR
            C[t, epi_i] = rho_t
            n = epi_i + 1
            w_t = rho_t / np.sum(C[t,:])
            X_t = (gamma ** t) * w_t * r
            X += np.clip(X_t, R_MIN, R_MAX)
            #print('X', X)
            # compute Y: estimate of action value function
            Y_t = (gamma**t)* w_t * Q_hat_pi_e[s, a]
            Y += np.clip(Y_t, R_MIN, R_MAX)
            #print('Y', Y)
            # compute Z: estimate of state value function
            if t == 0:
                # handle a corner case 
                w_t_min_1 = 1.0
            else:
                rho_t_min_1 = np.prod(rhos[:t])
                w_t_min_1 = rho_t_min_1 / n
            Z_T = (gamma ** t) * w_t_min_1 * V_hat_pi_e[s]
            Z += np.clip(Z_T, R_MIN, R_MAX)
            #print('Z', Z)
            # using D until epi_i, we compute DR and log it here
            DR_i += X - Y + Z
        #print('DR of {} at {}th episode'.format(DR_i, epi_i))
        DR.append(DR_i) # notice DR is just V_hat_pi_e return DR 

    return DR




def make_approximate_model_builder(num_states, num_actions):
    transition_count_table = np.zeros((num_states, num_actions, num_states))
    reward_sum_table = np.zeros((num_states, num_actions))
 
    def build_approximate_model(episode):
        # replay experiences and build N_sas and R_sa
        for s, a, r, new_s in episode:
           transition_count_table[s, a, new_s] += 1
           reward_sum_table[s, a] += r

        # build T_hat and R_hat
        transition_matrix = np.zeros((num_states, num_actions, num_states))
        reward_table = np.zeros((num_states, num_actions))

        for s in range(num_states):
            for a in range(num_actions):
                N_sa = np.sum(transition_count_table[s, a, :])
                if N_sa == 0:
                    # if never visited, no reward, no transition
                    continue
                reward_table[s, a] = reward_sum_table[s, a] / N_sa
                for new_s in range(num_states):
                    N_sas = transition_count_table[s, a, new_s]
                    transition_matrix[s, a, new_s] = N_sas / N_sa
        
        return transition_matrix, reward_table
    return build_approximate_model


def solve_MDP_equations(transition_matrix, R, GAMMA, policy_table):     
    
#    start = timer();
    
    STATE_COUNT = np.shape( transition_matrix )[ 0 ]
    ACTION_COUNT = np.shape( transition_matrix )[ 1 ]
    
    I = np.identity(STATE_COUNT);    
    M = np.zeros((STATE_COUNT,STATE_COUNT));
    b = np.zeros((STATE_COUNT))
    
    for s in range(STATE_COUNT):
        for a in range(ACTION_COUNT):
            p = policy_table[s,a]
            b[s] += p*np.dot(transition_matrix[s, a, :],R[:,a]);
            M[s,:] += p*transition_matrix[s, a, :];

    V = np.dot(np.linalg.inv(I-GAMMA*M),b)

    new_Q_table = np.zeros((STATE_COUNT, ACTION_COUNT))
    for a in range(ACTION_COUNT):
        new_Q_table[:,a] = np.dot(transition_matrix[:, a, :],(R[:,a]+GAMMA*V));

#    end = timer();
#    print('time elapsed in solve MDP eq is',end-start)
         
    return new_Q_table,V



def turn_policy_to_stochastic_policy(
        pi,
        num_of_states = None,
        num_of_actions = None ):
    if len(np.shape(pi)) == 1:
        if num_of_states is None:
            num_of_states = len(pi)
        if num_of_actions is None:
            num_of_actions = np.max(pi) + 1
            warnings.warn('number of actions not given. calculated number of actions : ' + str(num_of_actions))
        pi_original = np.copy(pi)
        pi = np.zeros( ( num_of_states, num_of_actions ) )
        for si in range(num_of_states):
            pi[si, pi_original[si] ] = 1
    return pi
            


def policy_evaluation( T , R , gamma, pi , theta = 0.1 ):
    
    start = timer()
    
    num_of_states = np.shape( T )[ 0 ]
    num_of_actions = np.shape( T )[ 1 ]
    # Evaluate V
    V = np.zeros( num_of_states )
    converged = False
    while not converged:
        delta = 0
        for si in range( num_of_states ):
            v = V[si]
            if pi.ndim == 2:
                V[si] = np.sum( np.tile( np.reshape( pi[si, :], (1, num_of_actions)), 
                    #(num_of_states, 1)) * T[ si, :, :].T * (R[si, :, :].T + 
                     (num_of_states, 1)) * T[ si, :, :].T * (R[si, :].T + 
                    gamma * np.tile( np.reshape( V, ( num_of_states, 1) ), ( 1, num_of_actions) ) ) ) 
            elif pi.ndim == 1:
                #V[si] = np.sum( T[ si, pi[si], : ] * ( R[ si, pi[si], : ] + gamma * V ) ) 
                V[si] = np.sum( T[ si, pi[si], : ] * ( R[ si, pi[si]] + gamma * V ) ) 
            delta = max( delta , abs( v - V[si] ) )
        converged = delta < theta
    # Evaluate Q
    Q = np.zeros( ( num_of_states, num_of_actions ) )
    for state_ind in range( num_of_states ):
        for action_ind in range( num_of_actions ):
            Q[ state_ind, action_ind ] = np.sum( T[ state_ind, action_ind, : ] * \
                #( R[ state_ind, action_ind, : ] + gamma * V ) )
                ( R[ state_ind, action_ind] + gamma * V ) )
    end = timer();
    print('time elapsed in policy_evaluation is',end-start)
         
    return Q,V


def off_policy_per_decision_doubly_robust(states_sequence, actions_sequence, rewards_sequence, fence_posts, trans_as_tuples, gamma,
                                          pi_evaluation, pi_behavior, V = None, Q = None):

    num_of_trials = len( fence_posts )
    individual_trial_estimators = []
    num_of_states = pi_behavior.shape[0]
    num_of_actions = pi_behavior.shape[1]
#    pi_evaluation = turn_policy_to_stochastic_policy( pi_evaluation, num_of_states = num_of_states, num_of_actions = num_of_actions )
#    pi_behavior = turn_policy_to_stochastic_policy( pi_behavior, num_of_states = num_of_states, num_of_actions = num_of_actions )
    # estimate V and Q if they are not passed as parameters
    
    build_approximate_model = make_approximate_model_builder(num_of_states, num_of_actions)
    
    if V is None or Q is None:
        
        T_hat, R_hat = build_approximate_model(trans_as_tuples)
        Q,V = policy_evaluation( T_hat , R_hat , gamma, pi_evaluation  )
    # calculate the doubly robust estimator of the policy
    for trial_i in range( num_of_trials ):
        current_trial_estimator = 0
        rho = 1
        discount = 1/gamma
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        for t in range(fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            previous_rho = rho
            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
            discount *= gamma
            current_trial_estimator += rho * discount * rewards_sequence[ t ] - \
                discount * ( rho * Q[ states_sequence[ t ], actions_sequence[ t ] ] - \
                             previous_rho * V[ states_sequence[ t ] ] )
        individual_trial_estimators += [ current_trial_estimator ]
    estimator = np.mean( individual_trial_estimators )

    return estimator, individual_trial_estimators



def off_policy_per_decision_weighted_doubly_robust(states_sequence, actions_sequence, rewards_sequence, fence_posts, trans_as_tuples, gamma,
                                                   pi_evaluation, pi_behavior, V = None, Q = None):

    num_of_trials = len( fence_posts )
    individual_trial_estimators = []
    num_of_states = pi_behavior.shape[0]
    num_of_actions = pi_behavior.shape[1]
#    pi_evaluation = turn_policy_to_stochastic_policy( pi_evaluation, num_of_states = num_of_states, num_of_actions = num_of_actions )
#    pi_behavior = turn_policy_to_stochastic_policy( pi_behavior, num_of_states = num_of_states, num_of_actions = num_of_actions )
    
    # Adding small quantities to pi behavior to avoid the division by zero:
    pi_behavior = pi_behavior + 1e-4
    print('sum(pi_evaluation < 0)',sum(pi_evaluation < 0))
    build_approximate_model = make_approximate_model_builder(num_of_states, num_of_actions)
#    print('pi_evaluation',pi_evaluation)
    # estimate V and Q if they are not passed as parameters
    if V is None or Q is None:
        T_hat, R_hat = build_approximate_model(trans_as_tuples)
        Q,V = solve_MDP_equations( T_hat, R_hat, gamma, pi_evaluation )
    # calculate the doubly robust estimator of the policy
    fence_posts_with_length_appended = fence_posts + [ len( states_sequence ) ]
    single_patient_sequences_length = [ fence_posts_with_length_appended[i+1] - \
        fence_posts_with_length_appended[i] for i in range(len(fence_posts)) ]
    length_of_longest_patient_sequence = max( single_patient_sequences_length )
    rho_array = np.nan * np.zeros( ( num_of_trials, length_of_longest_patient_sequence ) )
#    rho_array = np.ones( ( num_of_trials, length_of_longest_patient_sequence ) )

    # This loop computes the weights
    for trial_i in range( num_of_trials ):
        rho = 1
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        t_within_trial = 0
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):

            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
            
            # WARNING: CLIPPING RHO   
            rho=np.clip(rho, 0,3)
            
            rho_array[ trial_i, t_within_trial ] = rho
            t_within_trial += 1
        rho_array[ trial_i, t_within_trial: ] = rho
    weights_normalization = np.sum( rho_array, axis = 0 )
    
    # This loop computes the actual estimator
    for trial_i in range( num_of_trials ):
        current_trial_estimator = 0
        rho = 1
        w = 1 / num_of_trials
        discount = 1/gamma
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        t_within_trial = 0
#        print('fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial', fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial)
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
#            print('len states seq:',len(actions_sequence))
#            print('t',t)
            previous_w = w
            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
                
            # WARNING: CLIPPING RHO   
            rho=np.clip(rho, 0,3)
            
            w = rho / weights_normalization[ t_within_trial ]
            discount *= gamma
#            print('states_sequence[ t ], actions_sequence[ t ]:',states_sequence[ t ], actions_sequence[ t ])
#            print('shape Q, shape V',Q.shape,V.shape)
            current_trial_estimator += w * discount * rewards_sequence[ t ] - \
                discount * ( w * Q[ states_sequence[ t ], actions_sequence[ t ] ] - \
                             previous_w * V[ states_sequence[ t ] ] )
            t_within_trial += 1
        individual_trial_estimators += [ current_trial_estimator ]
    estimator = np.sum( individual_trial_estimators )

    return estimator, individual_trial_estimators

