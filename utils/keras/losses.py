
import tensorflow as tf
import numpy as np

from tensorflow.contrib.keras import backend as K


def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=1))

#
### Loss functions
#

def mean_euclidean(y_true, y_pred):
    return K.mean(euclidean_distance(y_true, y_pred))


def ms_euclidean(y_true, y_pred):
    return K.mean(K.sum(K.square(y_true - y_pred), axis=1))


def reg_mean_euclidean(y_true, y_pred):
    return K.mean(
        euclidean_distance(y_true, y_pred) * (
            1 +
            10 * euclidean_distance(y_true, tf.constant([0.0, 0.0]))
        )
    )


# def reg_mean_euclidean2(y_true, y_pred):
#     return K.mean(
#         (1-b)*euclidean_distance(y_true, y_pred) +
#         b*
#          * (
#             1 +
#             10 * euclidean_distance(y_true, tf.constant([0.0, 0.0]))
#         )
    # )
