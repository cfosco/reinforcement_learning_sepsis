"""
This file defines several useful custom layers (these are not actual keras.layers.Layer objects) to be used in the network
You can call them like this:
    `LayerName(...)(x)`

This is useful since very often you need to chain some layers (like the classical Conv + BatchNorm + NonLinearity)
"""

import numpy as np
import tensorflow as tf
import tqdm
from keras import backend as K
from keras.layers import Dense as kDense, PReLU, ELU, LeakyReLU, Activation, Permute, Conv2DTranspose, Conv1D as kConv1D, BatchNormalization, Add, Multiply, Dropout, Reshape, Lambda, \
    ZeroPadding1D
from config import BATCH_NORM


def Conv1D(filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation=None, momentum=0.9, training=None, BN=True, config=BATCH_NORM,
           use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
           activity_regularizer=None, kernel_constraint=None, bias_constraint=None, dropout=None, name=None, **kwargs):
    """conv -> BN -> activation"""

    def f(x):
        h = x
        if dropout is not None:
            h = Dropout(dropout)(h)
        if padding != "causal++":
            h = kConv1D(filters,
                        kernel_size,
                        strides=strides,
                        padding=padding,
                        dilation_rate=dilation_rate,
                        activation=None,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        **kwargs)(h)
        else:
            h = ZeroPadding1D(padding=(2, 0))(x)
            h = kConv1D(filters,
                        kernel_size,
                        strides=strides,
                        padding=None,
                        activation=None,
                        use_bias=use_bias,
                        dilation_rate=dilation_rate,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        **kwargs)(h)
            h = Lambda(lambda x_: x_[:, :-2, :])(h)
        h = _activation(activation, BN=BN, name=name, momentum=momentum, training=training, config=config)(h)
        return h

    return f


def Deconv1D(filters, kernel_size, strides=2, padding='same', dilation_rate=1, activation="prelu", momentum=0.9, BN=True, config=BATCH_NORM,
             use_bias=False, training=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
             activity_regularizer=None, kernel_constraint=None, bias_constraint=None, dropout=None, name=None):
    """`strides` is the upsampling factor"""

    def f(x):
        shape = list(x._keras_shape[1:])
        assert len(shape) == 2, "The input should have a width and a depth dimensions (plus the batch dimensions)"
        new_shape = shape[:-1] + [1] + [shape[-1]]
        h = Reshape(new_shape)(x)

        if dropout is not None:
            h = Dropout(dropout)(h)

        h = Conv2DTranspose(filters,
                            kernel_size,
                            strides=(strides, 1),
                            padding=padding,
                            dilation_rate=dilation_rate,
                            activation=None,
                            use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            activity_regularizer=activity_regularizer,
                            kernel_constraint=kernel_constraint,
                            bias_constraint=bias_constraint
                            )(h)
        h = _activation(activation, BN=BN, name=name, momentum=momentum, training=training, config=config)(h)
        shape = list(h._keras_shape[1:])
        new_shape = shape[:-2] + [filters]
        h = Reshape(new_shape)(h)
        return h

    return f


def ICNN_block(n_units_u, n_units_z, l, first_block, last_block, activation='relu'):
    """
    :param n_units_u: the dim of the hidden u_i
    :param n_units_z: the dim of the hidden z_i
    :param l: the id of the layer (to access the weights that should be kept positive easily)
    :param first_block: whether it is the first block or not (there is no z at the first block)
    :param last_block: whether it is the last block or not (there is no activation at the last block)
    :param activation: the actiivation function. You should stick with relu
    :param BN: whether to use batch normalization or not
    :return:
    """
    from keras.layers import Add, Multiply, Activation, Dense

    def f(u, z, a):
        # u_i+1 from u_i
        u_ = Dense(n_units_u, activation=activation, use_bias=True)(u)

        # z_i+1 from z_i and u_i and a
        # there are 3 blocks
        # First: dense on u
        z_u = Dense(n_units_z, activation=None, use_bias=True)(u)

        # Second: dense on u, then multiplied by the actions, then dense
        a_shape = a._keras_shape[1]
        z_au = Dense(a_shape, activation=None, use_bias=True)(u)
        z_au = Multiply()([a, z_au])
        z_au = Dense(n_units_z, activation=None, use_bias=False)(z_au)

        # Third: dense+relu on u, then multiplied by last z, then dense
        if not first_block:
            z_shape = z._keras_shape[1]
            z_zu = Dense(z_shape, activation='relu', use_bias=True)(u)
            z_zu = Multiply()([z_zu, z])
            z_zu = Dense(n_units_z, activation=None, use_bias=False, name='w' + str(l))(z_zu)  # these weights should stay positive

            # at last, sum them all
            z_ = Add()([z_u, z_au, z_zu])
            if not last_block:
                z_ = Activation(activation)(z_)
        else:
            z_ = Activation(activation)(Add()([z_au, z_u]))

        return u_, z_

    return f


# DENSE
def Dense(n_units, activation=None, BN=False, channel=1, training=None, config=BATCH_NORM, use_bias=True, **kwargs):
    def f(x):
        if len(x._keras_shape[1:]) == 2:
            if channel == 2:
                h = kDense(n_units, use_bias=use_bias, **kwargs)(x)
            elif channel == 1:
                h = Permute((2, 1))(kDense(n_units, use_bias=use_bias, **kwargs)(Permute((2, 1))(x)))
            else:
                raise ValueError('channel should be either 1 or 2')
            h = _activation(activation, BN=BN, training=training, config=config)(h)
            return h
        elif len(x._keras_shape[1:]) == 1:
            h = kDense(n_units, use_bias=use_bias, **kwargs)(x)
            return _activation(activation, BN=BN, training=training, config=config)(h)
        else:
            raise ValueError('len(x._keras_shape) should be either 2 or 3 (including the batch dim)')

    return f


# BatchNorm
def BatchNorm(momentum=0.99, training=True):
    def batchnorm(x, momentum=momentum, training=training):
        return tf.layers.batch_normalization(x, momentum=momentum, training=training)

    def f(x):
        return Lambda(batchnorm, output_shape=tuple([xx for xx in x._keras_shape if xx is not None]))(x)

    return f


# ACTIVATION
def _activation(activation, BN=True, name=None, momentum=0.9, training=None, config=BATCH_NORM):
    """
    A more general activation function, allowing to use just string (for prelu, leakyrelu and elu) and to add BN before applying the activation
    :param training: if using a tensorflow optimizer, training should be K.learning_phase()
                     if using a Keras optimizer, just let it to None
    """

    def f(x):
        if BN and activation != 'selu':
            if config == 'keras':
                h = BatchNormalization(momentum=momentum)(x, training=training)
            elif config == 'tf' or config == 'tensorflow':
                h = BatchNorm(is_training=training)(x)
            else:
                raise ValueError('config should be either `keras`, `tf` or `tensorflow`')
        else:
            h = x
        if activation is None:
            return h
        if activation in ['prelu', 'leakyrelu', 'elu']:
            if activation == 'prelu':
                return PReLU(name=name)(h)
            if activation == 'leakyrelu':
                return LeakyReLU(name=name)(h)
            if activation == 'elu':
                return ELU(name=name)(h)
        else:
            h = Activation(activation, name=name)(h)
            return h

    return f


def gate(x, num_filters):
    x1 = x[:, :, :num_filters // 2]
    x2 = x[:, :, num_filters // 2:]
    return Multiply()([x1, Activation('sigmoid')(x2)])


# AUTOREGRESSIVE MODELS
def PixelCNNGatedConv(kernel_size, num_filters, padding, activation="prelu", BN=True, residual=True, dropout=None, dilation_rate=1):
    """
    PixelCNN gated convolutions
    :param residual:
    :param dropout: dropout rate at each layer
    :param dilation_rate: an integer or tuple of two integers giving the dilation rates of the first and second convolutions
    :param kernel_size:
    :param num_filters:
    :param activation:
    :param BN:
    :param padding: `a` or `b`
                    `a` if it is the first conv (in this case the neuron at the same position isn't taken into account in the computation)
                    `b` if it's  not (the neuron at the same position is taken into account in the computation)

                    Examples with a kernel size of 3:
                    `a`:
                          _  <-- to compute this time step we use the three previous ones
                    _ _ _ x

                    `b`:
                          _  <-- to compute this time step we use the two previous one and the current one
                      _ _ _
    :return:
    """

    def f(x, h):
        if isinstance(dilation_rate, tuple):
            d1, d2 = dilation_rate
        else:
            d1 = d2 = dilation_rate

        # two causal convolutions (the first one being either type a or b, the second one being b)
        if padding == "b":  # access to the past and the present, but not the future
            xx = Conv1D(num_filters, kernel_size, padding="causal", BN=BN, use_bias=not BN, activation=activation, dropout=dropout, dilation_rate=d1)(x)
        elif padding == "a":  # shift to the right so that you only have access to the past
            xx = ZeroPadding1D(padding=(2, 0))(x)
            xx = Conv1D(num_filters, kernel_size, BN=BN, activation=activation, use_bias=not BN, dropout=dropout, dilation_rate=d1)(xx)
            xx = Lambda(lambda x_: x_[:, :-2, :])(xx)
        else:
            raise ValueError
        xx = Conv1D(num_filters * 2, kernel_size, padding="causal", BN=BN, use_bias=not BN, activation=None, dropout=dropout, dilation_rate=d2)(xx)

        # conditional vector
        h = Conv1D(num_filters * 2, 1, activation=activation, BN=BN, dropout=dropout)(h)
        xx = Add()([xx, h])

        # gate
        xx = Lambda(gate, arguments={"num_filters": 2 * num_filters})(xx)

        # add residual connexions if needed
        if residual:
            if padding == "b":  # access to the past and the present, but not the future
                x_ = x
            elif padding == "a":  # shift to the right so that you only have access to the past
                x_ = ZeroPadding1D(padding=(1, 0))(x)
                x_ = Lambda(lambda x_: x_[:, :-1, :])(x_)
            else:
                raise ValueError
            xx = Add()([xx, x_])
        return xx

    return f


def generate(model, z, original_dim, nchar, x=None, start=None):
    """Used to generate proteins with an autoregressive model (decoder using PixelCNNGatedConv)"""
    if x is None:
        x = np.zeros((z.shape[0], original_dim, nchar))
        start = 0
    for i in tqdm.tqdm(range(start, original_dim)):
        pred = model.predict([z, x])
        pred = pred[:, i, :]
        pred = pred.argmax(-1)
        for j, p in enumerate(pred):
            x[j, i, p] = 1
    return x


def WaveNet(kernel_size, num_filters, padding, dilation_rate=1, activation="prelu", BN=True, dropout=None):
    """
    Wavenet model. Takes as inputs a transformed temporal data and an up-sampled version of the latent code
    It outputs two tensors, each having the same dimensions as the input transformed temporal data
    :param padding: `a` or `b`. If `a`, use only inputs up to t-1. If `b`, up to t
    :param conv_on_h: whether to convolve on h before summing to x
    """

    def f(x):
        if isinstance(dilation_rate, tuple):
            d1, d2 = dilation_rate
        else:
            d1 = d2 = dilation_rate

        # two causal convolutions (the first one being either type a or b, the second one being b)
        if padding == "b":  # access to the past and the present, but not the future
            xx = Conv1D(num_filters, kernel_size, padding="causal", BN=BN, use_bias=not BN, activation=activation, dropout=dropout, dilation_rate=d1)(x)
        elif padding == "a":  # shift to the right so that you only have access to the past
            xx = ZeroPadding1D(padding=(2, 0))(x)
            xx = Conv1D(num_filters, kernel_size, BN=BN, activation=activation, use_bias=not BN, dropout=dropout, dilation_rate=d1)(xx)
            xx = Lambda(lambda x_: x_[:, :-2, :])(xx)
        else:
            raise ValueError
        xx = Conv1D(num_filters * 2, kernel_size, padding="causal", BN=BN, use_bias=not BN, activation=None, dropout=dropout, dilation_rate=d2)(xx)

        # gate
        xx = Lambda(gate, arguments={"num_filters": 2 * num_filters})(xx)

        xx = Conv1D(num_filters, 1, BN=BN, activation=activation)(xx)

        # create two outputs: out which will be passed to the next block and skip which will be summed to the others in the end
        skip = xx
        out = xx

        if padding == "b":  # access to the past and the present, but not the future
            x_ = x
        elif padding == "a":  # shift to the right so that you only have access to the past
            x_ = ZeroPadding1D(padding=(1, 0))(x)
            x_ = Lambda(lambda x_: x_[:, :-1, :])(x_)
        else:
            raise ValueError
        out = Add()([out, x_])

        return out, skip

    return f


def _mask_matrix_made(K, D):
    """A generator of masks for two-layered MADE model (see https://arxiv.org/pdf/1502.03509.pdf)"""
    mask_vector = np.random.randint(1, D, K)
    mask_matrix0 = np.fromfunction(lambda k, d: mask_vector[k] >= d, (K, D), dtype=int).astype(np.int32).astype(np.float32)
    mask_matrix1 = np.fromfunction(lambda d, k: d > mask_vector[k], (D, K), dtype=int).astype(np.int32).astype(np.float32)
    return mask_matrix0, mask_matrix1


def MADE(mask_matrix0, mask_matrix1, latent_dim):
    """A 2-layered MADE model (https://arxiv.org/pdf/1502.03509.pdf)"""

    def f(x):
        hl = MaskedDense(latent_dim, mask=mask_matrix0)(x)
        hl = PReLU()(hl)
        std = MaskedDense(latent_dim, mask=mask_matrix1, activation="softplus")(hl)
        mean = MaskedDense(latent_dim, mask=mask_matrix1, activation=None)(hl)
        return mean, std

    return f


class MaskedDense(kDense):
    """A dense layer with a masking possibilities"""

    def __init__(self, units, mask, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 transpose=False, **kwargs):
        super(MaskedDense, self).__init__(units, bias_initializer=bias_initializer,
                                          activation=activation, kernel_initializer=kernel_initializer,
                                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                          use_bias=use_bias, **kwargs)
        if not transpose:
            self.mask = K.variable(mask)
        else:
            self.mask = K.variable(mask.T)

    def call(self, x, mask=None):
        output = K.dot(x, Multiply()([self.kernel, self.mask]))
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output