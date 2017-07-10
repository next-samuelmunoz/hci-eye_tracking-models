# -*- coding: utf-8 -*-
"""Perform data augmentation on images
"""

from itertools import product

import numpy as np
import skimage
from skimage.restoration import denoise_bilateral
from skimage.util import random_noise
from skimage.exposure import equalize_hist


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


def bilateral(img):
    """Apply bilateral filter
    """
    return skimage.util.img_as_ubyte(
        denoise_bilateral(img, sigma_spatial=2, multichannel=True),
        force_copy=True
    )

def noise(img):
    """Add gaussian noise
    """
    return skimage.util.img_as_ubyte(
        random_noise(img, mode='gaussian', var=0.01),
        force_copy=True
    )

def equalize(img):
    """Equalize histogram
    """
    return skimage.util.img_as_ubyte(
        equalize_hist(img, nbins=256, mask=None),
        force_copy=True
    )
