import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

_MAX = 115.0
N = 75
N = tf.cast(N, tf.float32)
epsilon = 1e-5


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5 * _MAX / 2.0
    # return mean_squared_error(y_true, y_pred) ** 0.5 * std


def mape(y_true, y_pred):
    return tf.keras.losses.MAPE(y_true, y_pred)


def mae(y_true, y_pred):
    y_true = (y_true + 1.0) * _MAX / 2.0
    y_pred = (y_pred + 1.0) * _MAX / 2.0

    # y_true = y_true * std + mean
    # y_pred = y_pred * std + mean

    return tf.keras.losses.MAE(y_true, y_pred)


def smape(y_true, y_pred):
    return 2.0 * K.mean(tf.abs(y_pred - y_true) / (tf.abs(y_pred) + tf.abs(y_true) + epsilon)) * 100


def d_rmse(y_true, y_pred):
    y_true = K.sum(y_true, axis=1)
    y_pred = K.sum(y_pred, axis=1)

    return mean_squared_error(y_true, y_pred) ** 0.5 * _MAX / 2.0
    # eturn mean_squared_error(y_true, y_pred) ** 0.5 * std


def d_mape(y_true, y_pred):
    y_true = tf.reduce_sum(y_true, axis=1)
    y_pred = tf.reduce_sum(y_pred, axis=1)

    return tf.keras.losses.MAPE(y_true, y_pred)


def d_smape(y_true, y_pred):
    y_true = tf.reduce_sum(y_true, axis=1)
    y_pred = tf.reduce_sum(y_pred, axis=1)

    return 2.0 * K.mean(tf.abs(y_pred - y_true) / (tf.abs(y_pred) + tf.abs(y_true) + epsilon)) * 100


def d_mae(y_true, y_pred):
    y_true = (y_true + 1.0) * _MAX / 2.0
    y_pred = (y_pred + 1.0) * _MAX / 2.0
    # y_true = y_true * std + mean
    # y_pred = y_pred * std + mean

    y_true = tf.reduce_sum(y_true, axis=1)
    y_pred = tf.reduce_sum(y_pred, axis=1)

    return tf.keras.losses.MAE(y_true, y_pred)


def o_rmse(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, N, N))
    y_pred = tf.reshape(y_pred, (-1, N, N))

    y_true = K.sum(y_true, axis=-1)
    y_pred = K.sum(y_pred, axis=-1)

    return mean_squared_error(y_true, y_pred) ** 0.5 * _MAX / 2.0
    # return mean_squared_error(y_true, y_pred) ** 0.5 * std


def o_mape(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, N, N))
    y_pred = tf.reshape(y_pred, (-1, N, N))

    y_true = tf.reduce_sum(y_true, axis=-1)
    y_pred = tf.reduce_sum(y_pred, axis=-1)

    return tf.keras.losses.MAPE(y_true, y_pred)


def o_smape(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, N, N))
    y_pred = tf.reshape(y_pred, (-1, N, N))

    y_true = tf.reduce_sum(y_true, axis=-1)
    y_pred = tf.reduce_sum(y_pred, axis=-1)

    return 2.0 * K.mean(tf.abs(y_pred - y_true) / (tf.abs(y_pred) + tf.abs(y_true) + epsilon)) * 100


def o_mae(y_true, y_pred):
    y_true = (y_true + 1.0) * _MAX / 2.0
    y_pred = (y_pred + 1.0) * _MAX / 2.0
    # y_true = y_true * std + mean
    # y_pred = y_pred * std + mean

    y_true = tf.reshape(y_true, (-1, N, N))
    y_pred = tf.reshape(y_pred, (-1, N, N))

    y_true = tf.reduce_sum(y_true, axis=-1)
    y_pred = tf.reduce_sum(y_pred, axis=-1)

    return tf.keras.losses.MAE(y_true, y_pred)
