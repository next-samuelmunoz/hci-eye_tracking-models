# -*- coding: utf-8 -*-
"""Perform data augmentation on images
"""

from itertools import product

import numpy as np
import skimage
from skimage.restoration import denoise_bilateral
from skimage.util import random_noise
import skimage.filters
from skimage.filters import threshold_local


def data_augmentation(img, transformations=[], mirrored=False):
    """Iterate over transformations and return the transformed image.

    Parameters
    ----------
    img:
        Opened image with skimage.io.imread()
    transformations: function(img)
        Function to augment the image.
    mirrored: bool
        If img has suffered a mirror transformation.

    Returns
    -------
    img: generator of (transformed image, is mirrored).
    """
    if transformations==[]:  # Base case
        yield(img, mirrored)
    else:  # Apply first transformation
        t_apply, *t_list = transformations
        is_mirrored = mirrored or t_apply==mirror
        yield from data_augmentation(img, t_list, mirrored)
        yield from data_augmentation(
            t_apply(img),
            t_list,
            is_mirrored
        )


#
### Transformations
#

def mirror(img):
    """Vertical symmetry
    """
    return np.fliplr(img.copy())


def blur(img):
    """Apply bilateral filter
    """
    return np.clip(
        skimage.filters.gaussian(img,sigma=3.0, multichannel=True),
        -1,1
    ).copy()

def noise(img):
    """Add gaussian noise
    """
    return random_noise(img, mode='gaussian', var=0.001).copy()

def saltnpepper(img):
    """Add gaussian noise
    """
    return random_noise(img, mode='s&p', amount=0.05).copy()

def threshold(img):
    """Add gaussian noise
    """
    return threshold_local(img, block_size=5, method='gaussian', offset=0, mode='reflect', param=None).copy()
