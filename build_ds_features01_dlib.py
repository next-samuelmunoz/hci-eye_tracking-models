# -*- coding: utf-8 -*-
"""Build the features01 dataset with Dlib.
"""


import csv
import hashlib
import os

from skimage import io

import config
from utils.data import Data
from utils.features_dlib import FeaturesDlib
from utils.features01_dlib import dlib2features01, extract_eye


if __name__=="__main__":
    # Create destination path
    if os.path.exists(config.PATH_DATA_FEATURES01_DLIB):
        print("Folder {} exists, no need to generate dataset.".format(config.PATH_DATA_FEATURES01_DLIB))
        exit()
    os.makedirs(config.PATH_DATA_FEATURES01_DLIB)
    features = FeaturesDlib(config.PATH_DLIB)
    data = Data(config.PATH_DATA_RAW)
    i = 0
    with open(config.PATH_DATA_FEATURES01_DLIB_CSV, 'w') as fd:
        for datum in data.iterate():
            img_path = datum['img_path']
            try:
                img = io.imread(img_path)
                f = features.extract_features(img, config.THRESHOLD_FACE_WIDTH)
                f = dlib2features01(f)
                f.update(datum)
                f['img'] = '/'.join(img_path.split('/')[-2:])
                # Generate eyes
                eye_path = hashlib.md5(img_path.encode('utf-8')).hexdigest()
                for eye in ('eye_left','eye_right'):
                    f[eye+'_image'] = eye_path+'_'+eye+'.jpg'
                    img_eye = extract_eye(
                        img,
                        f[eye+'_x'], f[eye+'_y'],
                        f[eye+'_width'] ,f[eye+'_height']
                    )
                    io.imsave(
                        config.PATH_DATA_FEATURES01_DLIB+f[eye+'_image'],
                        img_eye
                    )
                if i==0:  # First case
                    csv_writer = csv.DictWriter(fd, fieldnames=f.keys())
                    csv_writer.writeheader()
                csv_writer.writerow(f)
                i+=1
            except Exception as e:
                print("[WARNING] Exception: {}, Image: {}".format(e, img_path))
