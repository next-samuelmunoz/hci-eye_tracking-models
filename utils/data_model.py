# -*- coding: utf-8 -*-
"""Data utilities to train models.
"""

import numpy as np
import pandas as pd
import sklearn.model_selection


def load(file_data, file_imgs_left, file_imgs_right): # TODO move params
    """Load preprocessed data from files.
    Returns
    -------
    _ : pandas DataFrame
    _ : images left array
    _ : images right array
    """
    return(
        pd.read_csv(file_data),
        np.load(file_imgs_left+".npy"),
        np.load(file_imgs_right+".npy")
    )


def split(data, imgs_left, imgs_right, train_size, validation_size, random_state=42):
    (  # Train - Test
        train_data, test_data,
        train_imgs_left, test_imgs_left,
        train_imgs_right, test_imgs_right
    ) = sklearn.model_selection.train_test_split(
        data, imgs_left, imgs_right,
        train_size=train_size,
        random_state=22
    )
    assert(len(train_data)==len(train_imgs_left)==len(train_imgs_right))
    assert(len(test_data)==len(test_imgs_left)==len(test_imgs_right))
    # Train - validation
    (
        train_data, validation_data,
        train_imgs_left, validation_imgs_left,
        train_imgs_right, validation_imgs_right
    ) = sklearn.model_selection.train_test_split(
        train_data, train_imgs_left, train_imgs_right,
        train_size=train_size,
        random_state=22
    )
    assert(len(train_data)==len(train_imgs_left)==len(train_imgs_right))
    assert(len(validation_data)==len(validation_imgs_left)==len(validation_imgs_right))
    return(
        (train_data, train_imgs_left, train_imgs_right),
        (validation_data, validation_imgs_left, validation_imgs_right),
        (test_data, test_imgs_left, test_imgs_right)
    )


def get_batch(data, imgs_left, imgs_right, batch_size):
    """Get a batch

    Parameters
    ----------
    data: pandas DataFrame
    imgs_left: left-eye images array
    imgs_right: right-eye images array
    batch_size: int

    Returns
    -------
    _ : np.array
    _ : np.array
    _ : np.array
    """
    index = np.arange(len(data))
    np.random.shuffle(index)  # Stochastic order
    data_random = data.iloc[index]
    imgs_left_random = imgs_left[index]
    imgs_right_random = imgs_right[index]
    for i in range(0, len(data), batch_size):
        yield(
            data_random[i:i+batch_size],
            imgs_left_random[i:i+batch_size],
            imgs_right_random[i:i+batch_size]
        )
