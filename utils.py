# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf

import config


def load_data():
    """Load preprocessed data from files.
    Returns
    -------
    _ : pandas DataFrame
    _ : images dictionary
    """
    return(
        pd.read_csv(config.PATH_DATA),
        np.load(config.PATH_IMGS)
    )


def get_batch(data, imgs, batch_size):
    """Get a batch

    Parameters
    ----------
    data: pandas DataFrame
    imgs: images dictionary
    batch_size: int

    Returns
    -------
    _ : pandas Dataframe
    _ : left eye images
    _ : right eye images
    """
    assert(len(data)*2<=len(imgs.files))
    index = np.arange(len(data))
    np.random.shuffle(index)
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch_imgs_left = np.array([imgs[path] for path in batch_data['eye_left_image']])
        batch_imgs_right = np.array([imgs[path] for path in batch_data['eye_right_image']])
        yield(batch_data, batch_imgs_left, batch_imgs_right)


def model_save(session, name):
    saver = tf.train.Saver()
    save_path = saver.save(session, "data/models/"+name)


def model_load(session, name):
    saver = tf.train.Saver()
    saver.restore(session, "data/models/"+name)
