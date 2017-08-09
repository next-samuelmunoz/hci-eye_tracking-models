# -*- coding: utf-8 -*-
"""Build the augmented features01 dataset with Dlib.
"""
import traceback

import csv
import hashlib
import os
import traceback

from skimage import io
import skimage.color
import skimage.util

import config
from utils.data import Data
from utils import data_augmentation
from utils.features_dlib import FeaturesDlib
from utils.features02_dlib import dlib2features, extract_eye


if __name__=="__main__":
    # Create destination path
    if os.path.exists(config.PATH_DATA_FEATURES02_DLIB_AUGMENTED):
        print("Folder {} exists, no need to generate dataset.".format(config.PATH_DATA_FEATURES02_DLIB_AUGMENTED))
        exit()
    os.makedirs(config.PATH_DATA_FEATURES02_DLIB_AUGMENTED)
    features = FeaturesDlib(config.PATH_DLIB)
    data = Data(config.PATH_DATA_RAW)
    i = 0
    with open(config.PATH_DATA_FEATURES02_DLIB_AUGMENTED_CSV, 'w') as fd:
        for datum in data.iterate():
            img_path = datum['img_path']
            img_original = skimage.color.rgb2gray(io.imread(img_path)) # Work with a grayscale img
            i_transform = 0
            # Data augmentation
            for (img, mirrored) in data_augmentation.data_augmentation(
                img_original,
                transformations=[  # Expensive operations first
                    data_augmentation.blur,
                    data_augmentation.threshold,
                    data_augmentation.saltnpepper,
                    data_augmentation.noise,
                    data_augmentation.mirror,
                ]
            ):
                print("IMG: {}\t\t Transformation: {}".format(i, i_transform))
                try:
                    f = features.extract_features(
                        skimage.util.img_as_ubyte(img),
                        config.THRESHOLD_FACE_WIDTH
                    )
                    f = dlib2features(f)
                    f.update(datum)
                    f['img'] = '/'.join(img_path.split('/')[-2:])
                    # Generate eyes
                    eye_path = hashlib.md5(img_path.encode('utf-8')).hexdigest()
                    for eye in ('eye_left','eye_right'):
                        f[eye+'_image'] = eye_path+'_'+eye+"_"+str(i_transform)+'.jpg'
                        img_eye = extract_eye(
                            img,
                            f[eye+'_x'], f[eye+'_y'],
                            f[eye+'_width'] ,f[eye+'_height'],
                            config.FEATURES02_EYES_SIZE
                        )
                        io.imsave(
                            config.PATH_DATA_FEATURES02_DLIB_AUGMENTED+f[eye+'_image'],
                            img_eye
                        )
                    # Mirror x target label if transformation is mirrored
                    if mirrored:
                        f['x'] = -f['x']+f['screen_width']
                    # To CSV
                    if i==0 and i_transform==0:  # First case
                        csv_writer = csv.DictWriter(fd, fieldnames=f.keys())
                        csv_writer.writeheader()
                    csv_writer.writerow(f)
                except Exception as e:
                    print("[WARNING] Exception: {}, Image: {}".format(e, img_path))
                    # traceback.print_exc()
                i_transform += 1
            i+=1
