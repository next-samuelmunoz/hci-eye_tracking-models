# -*- coding: utf-8 -*-
"""Perform data augmentation on images
"""

from itertools import product

import numpy as np
import skimage
from skimage.restoration import denoise_bilateral
from skimage.util import random_noise
from skimage.exposure import equalize_hist


def data_augmentation(img, transformations=[]):
    """Iterate over transformations and return the transformed image.

    Parameters
    ----------
    img:
        Opened image with skimage.io.imread()
    transformations: function(img)
        Function to augment the image.

    Returns
    -------
    img: generator of (transformed image, is mirrored).
    """
    i_mirror = transformations.index(mirror)
    for sequence in product([0,1], repeat=len(transformations)):
        transformation = [  # List of transformations to use
            t
            for t,use in zip(transformations, sequence)
            if use==1
        ]
        t_img = img.copy()
        for func in transformation:  # Apply transformations
            t_img = func(t_img)
        is_mirrored = sequence[i_mirror]==1
        yield (t_img, is_mirrored)


#
### Transformations
#

def mirror(img):
    """Vertical symmetry
    """
    return np.fliplr(img)


def bilateral(img):
    """Apply bilateral filter
    """
    return skimage.util.img_as_ubyte(
        denoise_bilateral(img, sigma_spatial=2, multichannel=True),
        force_copy=False
    )

def noise(img):
    """Add gaussian noise
    """
    return skimage.util.img_as_ubyte(
        random_noise(img, mode='gaussian', var=0.01),
        force_copy=False
    )

def equalize(img):
    """Equalize histogram
    """
    return skimage.util.img_as_ubyte(
        equalize_hist(img, nbins=256, mask=None),
        force_copy=False
    )
