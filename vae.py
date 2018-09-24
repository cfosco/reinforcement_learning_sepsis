"""
Define the two parts of the variational autoencoder: the encoder and the decoder (or generator)
"""
from keras.layers import Input, Concatenate, Flatten, Add, Reshape, Lambda, ZeroPadding1D
from models_keras import Conv1D, _mask_matrix_made, MADE, MaskedDense, WaveNet, _activation, Dense
from keras.models import Model
from parse_dataset import binary_columns, binary_actions_columns, binary_features_columns, numerical_columns_not_to_be_logged, numerical_columns_to_be_logged


def encoder(num_filters=10, kernel_size=2, T=32, n_num_features=len(numerical_columns_not_to_be_logged + numerical_columns_to_be_logged), latent_dim=32, BN=True, activation="prelu", dropout=.1, n_binary_features=len(binary_features_columns), n_binary_actions=len(binary_actions_columns)):
    """

    :param n_binary_actions:
    :param n_binary_features:
    :param num_filters: the number of filters to be used in the convolutions
    :param kernel_size: the size of the convolution kernel
    :param T: the length of each history
    :param n_num_features: the number of vitals/demographics/lab values per patient
    :param latent_dim: dimension of the latent representation
    :param BN: whether to use batch normalization or not
    :param activation: the activation to be used as a str. See doc of models.Dense
    :param dropout: None or a float between 0 and 1
    :return: a model that takes as inputs the vitals and the actions taken at each time step, and outputs the parameters
             of a Gaussian distribution (the mean and the variance)
    """
    binary_features = Input((n_binary_features, ))  # these are features that shouldn't be predicted (conditions). They don't depend on time...
    binary_features_ = Concatenate(1)(T*[Reshape((1, -1))(binary_features)])
    binary_actions = Input((T, n_binary_actions))  # these are actions that shouldn't be predicted (conditions)
    vitals = Input((T, n_num_features))
    actions = Input((T, 2))  # for each time step, two values: the VP and the IV. # including the actions is necessary: it is part of the history. Architecturally, it corresponds to a CVAE

    x = Concatenate(-1)([vitals, binary_features_, binary_actions, actions])
    h = Conv1D(num_filters, kernel_size, activation=activation, BN=BN, dropout=dropout, padding='causal')(x)
    h = Conv1D(num_filters, kernel_size, activation=activation, BN=BN, dropout=dropout, padding="causal",
               dilation_rate=2)(h)
    h = Conv1D(num_filters, kernel_size, activation=activation, BN=BN, dropout=dropout, padding="causal",
               dilation_rate=4)(h)
    h = Conv1D(num_filters, kernel_size, activation=activation, BN=BN, dropout=dropout, padding="causal",
               dilation_rate=8)(h)
    h = Conv1D(1, 1, activation=activation, BN=BN, dropout=dropout)(h)
    h = Flatten()(h)

    mask0, mask1 = _mask_matrix_made(latent_dim, latent_dim)
    z_mean, z_std = MADE(mask0, mask1, latent_dim)(h)

    E = Model([vitals, binary_features, binary_actions, actions], [z_mean, z_std])
    return E


def encoder2(num_filters=10, kernel_size=2, T=32, n_num_features=len(numerical_columns_not_to_be_logged + numerical_columns_to_be_logged), latent_dim=32, BN=True, activation="prelu", dropout=.1, n_binary_features=len(binary_features_columns), n_binary_actions=len(binary_actions_columns)):
    """

    :param n_binary_actions:
    :param n_binary_features:
    :param num_filters: the number of filters to be used in the convolutions
    :param kernel_size: the size of the convolution kernel
    :param T: the length of each history
    :param n_num_features: the number of vitals/demographics/lab values per patient
    :param latent_dim: dimension of the latent representation
    :param BN: whether to use batch normalization or not
    :param activation: the activation to be used as a str. See doc of models.Dense
    :param dropout: None or a float between 0 and 1
    :return: a model that takes as inputs the vitals and the actions taken at each time step, and outputs the parameters
             of a Gaussian distribution (the mean and the variance)
    """
    binary_features = Input((n_binary_features, ))  # these are features that shouldn't be predicted (conditions). They don't depend on time...
    binary_features_ = Concatenate(1)(T*[Reshape((1, -1))(binary_features)])
    binary_actions = Input((T, n_binary_actions))  # these are actions that shouldn't be predicted (conditions)
    vitals = Input((T, n_num_features))
    actions = Input((T, 2))  # for each time step, two values: the VP and the IV. # including the actions is necessary: it is part of the history. Architecturally, it corresponds to a CVAE

    x = Concatenate(-1)([vitals, binary_features_, binary_actions, actions])
    h = Conv1D(num_filters, 1, activation=activation, BN=BN, dropout=dropout)(x)
    h = Conv1D(num_filters, kernel_size, activation=activation, BN=BN, dropout=dropout, padding='causal')(h)
    h = Conv1D(num_filters, 1, activation=activation, BN=BN, dropout=dropout)(h)
    h = Conv1D(num_filters, kernel_size, activation=activation, BN=BN, dropout=dropout, padding="causal",
               dilation_rate=2)(h)
    h = Conv1D(num_filters, 1, activation=activation, BN=BN, dropout=dropout)(h)
    h = Conv1D(num_filters, kernel_size, activation=activation, BN=BN, dropout=dropout, padding="causal",
               dilation_rate=4)(h)
    h = Conv1D(num_filters, 1, activation=activation, BN=BN, dropout=dropout)(h)
    h = Conv1D(num_filters, kernel_size, activation=activation, BN=BN, dropout=dropout, padding="causal",
               dilation_rate=8)(h)
    h = Conv1D(1, 1, activation=activation, BN=BN, dropout=dropout)(h)
    h = Flatten()(h)

    mask0, mask1 = _mask_matrix_made(latent_dim, latent_dim)
    z_mean, z_std = MADE(mask0, mask1, latent_dim)(h)

    E = Model([vitals, binary_features, binary_actions, actions], [z_mean, z_std])
    return E


def generator(n_gates=3, T=32, n_num_features=len(numerical_columns_not_to_be_logged + numerical_columns_to_be_logged), latent_dim=32, kernel_size=2, activation="prelu", BN=True, dropout=.1, n_binary_features=len(binary_features_columns), n_binary_actions=len(binary_actions_columns)):
    """
    :param n_gates: the number of WaveNet blocks to be stacked. With 3 of them the receptive field is 32
    :param T: the duration of histories
    :param n_num_features: the number of vitals/demographics/lab values per time step
    :param latent_dim: the dimension of the latent code
    :param num_filters: the number of filters used in WaveNet blocks
    :param kernel_size: the size of the used kernels
    :param activation: the activation function used (see )
    :param BN: whether to use Batch normalization or not
    :param dropout: None or the % of nodes to be randomly dropped at each layer
    :return: a model taking as inputs (latent_code, vitals, actions) and outputs (next_vitals, reward)
    """
    # UPSAMPLING OF LATENT CODE
    latent_vector = Input((latent_dim,))
    mask0, mask1 = _mask_matrix_made(latent_dim, latent_dim)
    h_latent = MaskedDense(latent_dim, mask0)(latent_vector)
    h_latent = _activation(activation, BN=False)(h_latent)
    h_latent = MaskedDense(latent_dim, mask1)(h_latent)
    h_latent = Reshape((-1, 1))(h_latent)

    # CREATE AN INPUT FOR THE TEMPORAL DATA
    vitals = Input((T, n_num_features))  # numerical values
    binary_features = Input((n_binary_features,))  # these are features that shouldn't be predicted (conditions). They don't depend on time...
    binary_features_ = Concatenate(1)(T * [Reshape((1, -1))(binary_features)])
    binary_actions = Input((T, n_binary_actions))  # binary values (gender, rrt, sedation, ...)
    actions = Input((T, 2))  # for each time step, two values: the VP and the IV. Can be continuous or discrete
    temporals = Conv1D(n_num_features, kernel_size=1, activation=activation, BN=BN, dropout=dropout)(
        Concatenate(-1)([vitals, binary_features_, binary_actions, actions, h_latent]))

    # STACKED WAVENET BLOCKS
    temporals, skip = WaveNet(kernel_size, n_num_features, "a", dropout=dropout, dilation_rate=(1, 2))(temporals)
    skip_connections = [skip]

    for k in range(n_gates - 1):
        temporals, skip = WaveNet(kernel_size, n_num_features, "b", dropout=dropout,
                                  dilation_rate=(2 ** (2 * k + 2), 2 ** (2 * k + 3)))(temporals)
        skip_connections.append(skip)

    temporals = Add()(skip_connections)
    temporals = Conv1D(n_num_features, 1, activation=activation, BN=BN, dropout=dropout)(temporals)

    # for the numerical values, predict the difference from the previous state, to learn relative rather than absolute target. Hope that it can generalize better
    switched_vitals = Lambda(lambda x: x[:, :-1, :])(ZeroPadding1D(padding=(1, 0))(vitals))  # basically shift the input sequence to the right
    delta_vitals = Conv1D(n_num_features, 1, activation=None, BN=BN, dropout=None)(temporals)
    next_vitals = Add()([switched_vitals, delta_vitals])

    finished = Flatten()(Conv1D(1, 1, activation="sigmoid", BN=BN, dropout=None)(temporals))  # whether it is the end of the sequence or not. Might not be needed ???
    dead_or_alive = Flatten()(Conv1D(1, 1, activation="sigmoid", BN=BN, dropout=None)(temporals))  # whether the patient is alive (1) or not (0)
    # finished = Dense(1, activation="sigmoid")(Flatten()(temporals))  # whether it is the end of the sequence or not. Might not be needed ???
    # dead_or_alive = Dense(1, activation="sigmoid")(Flatten()(temporals))  # whether the patient is alive (1) or not (0)
    G = Model([vitals, binary_features, binary_actions, actions, latent_vector], [next_vitals, finished, dead_or_alive])
    return G