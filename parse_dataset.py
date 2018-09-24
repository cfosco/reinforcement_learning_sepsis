"""
This file contains functions that are used as preprocessing steps of the features of the MIMIC dataset
They modify the pandas dataframe containing the data
See one of the notebook to have a working example
"""
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


binary_columns = ['gender',
                  're_admission',
                  'rrt',
                  'mechvent',
                  'sedation'
                  ]
binary_actions_columns = [
    'rrt',
    'mechvent',
    'sedation'
]
binary_features_columns = [
    'gender',
    're_admission'
]
numerical_columns_to_be_logged = ['age',
                                  'Weight_kg',
                                  'SpO2',
                                  'SGOT',
                                  'SGPT',
                                  'Total_bili',
                                  'WBC_count',
                                  'Platelets_count',
                                  'PTT',
                                  'PT',
                                  'INR',
                                  'paO2',
                                  'paCO2',
                                  'Arterial_lactate',
                                  'PaO2_FiO2',
                                  # 'input_total_tev',
                                  'output_total'
                                  ]  # 'cumulated_balance_tev']
numerical_columns_not_to_be_logged = ['SIRS',
                                      'elixhauser',
                                      'SOFA',
                                      'FiO2_1',
                                      'GCS',
                                      'HR',
                                      'SysBP',
                                      'Arterial_BE',
                                      'MeanBP',
                                      'DiaBP',
                                      'Shock_Index',
                                      'RR',
                                      'SpO2',
                                      'Temp_C',
                                      'Potassium',
                                      'Sodium',
                                      'Chloride',
                                      'Glucose',
                                      'BUN',
                                      'Creatinine',
                                      'Magnesium',
                                      'Calcium',
                                      'Ionised_Ca',
                                      'CO2_mEqL',
                                      'Albumin',
                                      'Hb',
                                      'Arterial_pH',
                                      'Arterial_BE',
                                      'HCO3']
action_cols = ['max_dose_vaso', 'input_4hourly_tev']
log_action_cols = ['log_vaso', 'log_fluid']


def split_train_test_idx(data, train_prop=.95):
    ids = shuffle(data.icustayid.unique())
    return ids[:int(train_prop * ids.shape[0])], ids[int(train_prop * ids.shape[0]):]


def compute_action_quantiles(data):
    quantiles_fluid = {q: np.percentile(data.input_4hourly_tev.loc[data.input_4hourly_tev > 0].apply(np.log), q) for q
                       in [25, 50, 75]}
    quantiles_vaso = {q: np.percentile(data.max_dose_vaso.loc[data.max_dose_vaso > 0].apply(np.log), q) for q in
                      [25, 50, 75]}
    return quantiles_fluid, quantiles_vaso


def quantized_actions(fluid, vaso, quantiles_fluid, quantiles_vaso):
    """Divide the possible actions in 25 categories"""
    if fluid == 0:
        i = 0
    else:
        if np.log(fluid) <= quantiles_fluid[25]:
            i = 1
        if quantiles_fluid[25] < np.log(fluid) <= quantiles_fluid[50]:
            i = 2
        if quantiles_fluid[50] < np.log(fluid) <= quantiles_fluid[75]:
            i = 3
        if quantiles_fluid[75] < np.log(fluid):
            i = 4
    if vaso == 0:
        j = 0
    else:
        if np.log(vaso) <= quantiles_vaso[25]:
            j = 1
        if quantiles_vaso[25] < np.log(vaso) <= quantiles_vaso[50]:
            j = 2
        if quantiles_vaso[50] < np.log(vaso) <= quantiles_vaso[75]:
            j = 3
        if quantiles_vaso[75] < np.log(vaso):
            j = 4
    return i + 5 * j


def create_action_column(data):
    quantiles_fluid, quantiles_vaso = compute_action_quantiles(data)
    actions = []
    for t in data[['input_4hourly_tev', 'max_dose_vaso']].itertuples():
        fluid = t.input_4hourly_tev
        vaso = t.max_dose_vaso
        actions.append(quantized_actions(fluid, vaso, quantiles_fluid, quantiles_vaso))

    data['action'] = pd.Series(actions, index=data.index)
    data.action = data.action.apply(int)


def add_small_quantities(data):
    # adding small quantities to each zeros logged value. These values were chosen to be 50-100 times smaller than the lowest nonzero value
    data.Total_bili += 1e-4
    data.SGOT += 1e-2
    data.SGPT += 1e-2
    data.input_total_tev += 1e-3
    data.output_total += 1e-2
    data.PaO2_FiO2 += 1e-1
    data.Arterial_lactate += 1e-3
    data.PTT += 1e-1
    data.INR += 1e-3
    data.paCO2 += 1e-1
    data.paO2 += 1e-1


def add_relative_time_column(data):
    # add a column containing the time in hours since arrival
    relative_times = []
    patients = set()
    for t in data[['icustayid', 'charttime']].itertuples():
        time = t.charttime
        patient = t.icustayid
        if patient not in patients:
            start_time = time
            patients.add(patient)
        relative_times.append((time - start_time) / 3600)
    data['relative_time'] = pd.Series(relative_times, index=data.index)
    return data


def drop_patients_with_unrealistic_HR_or_BP(data):
    patients_to_drop = data[(data.HR == 8) | (data.DiaBP < 0)].icustayid.unique()
    data = data.loc[~data.icustayid.isin(patients_to_drop)]
    return data


def replace_absurd_temperatures(data):
    data.loc[data.Temp_C < 33, 'Temp_C'] = 37


def drop_patients_with_absurd_weights(data):
    return data.loc[~data.icustayid.isin(data.loc[data.Weight_kg < 20].icustayid.unique())]


def drop_patient_with_negative_input(data):
    return data.loc[~data.icustayid.isin(data.loc[data.input_total_tev < 0].icustayid.unique())]


def add_log_actions(data):
    eps_vaso = data.loc[data.max_dose_vaso>0][['max_dose_vaso']].values.min()
    eps_fluid = data.loc[data.input_4hourly_tev>0][['input_4hourly_tev']].values.min()
    data['log_vaso'] = data.max_dose_vaso.apply(lambda x: np.log(x*(x>0) + (x==0)*eps_vaso))
    data['log_fluid'] = data.input_4hourly_tev.apply(lambda x: np.log(x*(x>0) + (x==0)*eps_fluid))


def matrify_histories(data, idx, scaler, log_scaler, action_scaler, T=25, verbose=True, log_action=True):
    """
    Transform the Pandas dataset into numpy arrays that can be used to train a neural network
    :param data: a dataframe containing the sepsis dataset
    :param idx: the values of `icustayid` we want to filter on (typically to extract a train and a test set)
    :param scaler: the sklearn.preprocessing.StandardScaler that scales the numerical values
    :param log_scaler: the sklearn.preprocessing.StandardScaler that scales the numerical logged values
    :param verbose: whether to print stuff or not
    :return: several 3d arrays, with the first dimension (the number of patients selected by idx)
                - X_bin: the values of the binary columns, X_num, X_action, X_finished, X_alive
                - X_num: the numerical values
                - X_action: the action that were taken
                - X_finished : an array containing a 1 at the final timestep
                - X_alive: an array that is 1 while the patient is alive and gets 0 at the end
    """
    n_cols = len(binary_columns) + len(numerical_columns_not_to_be_logged) + len(numerical_columns_to_be_logged)
    n = idx.shape[0]

    X_action_bin = np.zeros((n, T, len(binary_actions_columns)))  # rrt, sedation, mechvent
    X_features_bin = np.zeros((n, len(binary_features_columns)))  # gender, re_admission
    X_num = np.zeros((n, T, n_cols - len(binary_columns)))
    X_action = np.zeros((n, T, 2))
    X_finished = np.zeros((n, T, ))
    X_alive = np.zeros((n, T, ))

    if verbose:
        iterator = tqdm(enumerate(data.loc[data.icustayid.isin(idx)].groupby('icustayid')))
    else:
        iterator = enumerate(data.loc[data.icustayid.isin(idx)].groupby('icustayid'))
    for k, (idx, df) in iterator:
        x_action_bin = df[binary_actions_columns].values
        x_features_bin = df[binary_features_columns].values[0]
        x_num_not_logged = scaler.transform(df[numerical_columns_not_to_be_logged])
        x_num_logged = log_scaler.transform(np.log(df[numerical_columns_to_be_logged]))
        x_num = np.concatenate([x_num_not_logged, x_num_logged], -1)
        if log_action:
            actions = action_scaler.transform(df[log_action_cols].values)
        else:
            actions = action_scaler.transform(df[action_cols].values)
        finished = np.zeros((T, ))
        finished[len(df) - 1:] = 1
        alive = np.ones((T, ))
        if df[['mortality_90d']].iloc[0].values[0] == 1:
            # alive[T-1:] = 0
            if df[['died_in_hosp']].iloc[0].values[0] == 1:
                alive[len(df) - 1:] = 0
            else:
                alive[len(df):] = 0

        # pad with zeros
        x_action_bin = np.concatenate([x_action_bin, np.zeros((T - len(df), len(binary_actions_columns)))], 0)
        x_num = np.concatenate([x_num, np.zeros((T - len(df), n_cols - len(binary_columns)))], 0)
        actions = np.concatenate([actions, np.zeros((T - len(df), 2))], 0)

        # k-th history
        X_action_bin[k] = x_action_bin
        X_features_bin[k] = x_features_bin
        X_num[k] = x_num
        X_action[k] = actions
        X_finished[k] = finished
        X_alive[k] = alive

    return     


def transition_iterator(data, idx=None, scaler=StandardScaler(), log_scaler=StandardScaler(), action_scaler=StandardScaler(),  RMAX=15, log_action=True):
    """
    Put the dataset in form of a list of transition (s,a,r,s')
    :param data: the pandas dataframe containing the sepsis dataset
    :param idx: the icustayids of the patients you want to consider (for example for split into train and test set). If None, returns the entire dataset
    :param scaler: (sklearn.preprocessing.StandardScaler)
                    the scaler of the numerical features that are not logged. It should be already fitted
    :param log_scaler: (sklearn.preprocessing.StandardScaler)
                        the scaler of the logged numerical features. It should already be fitted
    :param action_scaler: (sklearn.preprocessing.StandardScaler)
                           the scaler of the actions. It should already be fitted
    :param RMAX: the max reward
    :param log_action: whether to extract take the logged or the normal actions
    :return:
    """
    TRANSITIONS = []
    
    if idx is None:
        iterator = enumerate(data.groupby('icustayid'))
    else:   
        iterator = enumerate(data.loc[data.icustayid.isin(idx)].groupby('icustayid'))
    for k, (idx, df) in iterator:
        # create the state: concatenation of numerical values (logged or not logged), the binary actions, and the binary features (gender, re_admission)
        state_action_bin = df[binary_actions_columns].values
        state_features_bin = df[binary_features_columns].values
        state_num_not_logged = scaler.transform(df[numerical_columns_not_to_be_logged])
        state_num_logged = log_scaler.transform(np.log(df[numerical_columns_to_be_logged]))
        state = np.concatenate([state_num_not_logged, state_num_logged, state_action_bin, state_features_bin], -1)

        # actions
        if log_action:
            actions = action_scaler.transform(df[log_action_cols].values)
        else:
            actions = action_scaler.transform(df[action_cols].values)

        # rewards: 0 always, and final +-RMAX depending on the
        rewards = np.zeros((len(df), ))
#        if df[['mortality_90d']].iloc[0].values[0] == 1:
#            if df[['died_in_hosp']].iloc[0].values[0] == 1:
#                rewards[-1] = -RMAX
#            else:
#                rewards[-1] = RMAX

        if df[['mortality_90d']].iloc[0].values[0] == 1:
            rewards[-1] = -RMAX
        else:
            rewards[-1] = RMAX 

        # add transitions to the list
        for k in range(len(df) - 1):
            TRANSITIONS.append((state[k], actions[k], rewards[k], state[k+1]))
        TRANSITIONS.append((state[-1], actions[-1], rewards[-1], None))

    return TRANSITIONS
