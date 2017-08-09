# -*- coding: utf-8 -*-
"""Eye tracker predictor
"""

from tensorflow.contrib import keras
import numpy as np
import skimage.util

import config
from utils.features_dlib import FeaturesDlib
from utils.features02_dlib import dlib2features, extract_eye, FEATURES
from utils.normalize2 import normalize_data, normalize_features02

from . import losses


class Predictor:

    def __init__(self,
        path_dlib_model, model_name,
        screen_width, screen_height,
        webcam_width, webcam_height,
        threshold_face_width
    ):
        self.dlib_model = FeaturesDlib(path_dlib_model)  # Face landmarks
        self.model = keras.models.load_model(  # Pretrained model
            filepath=config.PATH_MODELS_KERAS+model_name,
            custom_objects={
                "mean_euclidean": losses.mean_euclidean,
                "ms_euclidean": losses.ms_euclidean,
                "reg_mean_euclidean": losses.reg_mean_euclidean
            }
        )
        # Constants
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.WEBCAM_WIDTH = webcam_width
        self.WEBCAM_HEIGHT = webcam_height
        self.THRESHOLD_FACE_WIDTH = threshold_face_width


    def _img2features(self, img):
        # Detect Landmarks
        f = self.dlib_model.extract_features(
            skimage.util.img_as_ubyte(img),
            self.THRESHOLD_FACE_WIDTH
        )
        landmarks = dlib2features(f)
        # Generate eye arrays
        eyes = {}
        for eye in ('eye_left','eye_right'):
            eyes[eye] = extract_eye(
                img,
                landmarks[eye+'_x'], landmarks[eye+'_y'],
                landmarks[eye+'_width'] ,landmarks[eye+'_height'],
                config.FEATURES02_EYES_SIZE
            )
        # Normalize data -> [-1,1]
        normalize_features02(landmarks, FEATURES, self.WEBCAM_WIDTH, self.WEBCAM_HEIGHT)
        return(
            eyes['eye_left'].reshape(1, *eyes['eye_left'].shape),
            eyes['eye_right'].reshape(1, *eyes['eye_right'].shape),
            np.array([[landmarks[x] for x in FEATURES]])
        )


    def predict(self, img):
        try:
            left_img, right_img, features = self._img2features(img)
            x,y = self.model.predict(
                x={
                    'left_imgs': left_img,
                    'right_imgs': right_img,
                    'features': features
                })[0]
            # Rescale to screen coordinates
            x,y = (x+1)/2*self.SCREEN_WIDTH, (y+1)/2*self.SCREEN_HEIGHT
            # Bound prediction to limits
            x = x if x>=0 else 0
            x = x if x<=self.SCREEN_WIDTH else self.SCREEN_WIDTH
            y = y if y>=0 else 0
            y = y if y<=self.SCREEN_HEIGHT else self.SCREEN_HEIGHT
        except Exception as e:
            raise e
        return (x,y)


    def get_error(self, img, target):
        try:
            left_img, right_img, features = self._img2features(img)
            retval = self.model.evaluate(
                x={
                    'left_imgs': left_img,
                    'right_imgs': right_img,
                    'features': features
                },
                y= np.array([target]),
                verbose=False,
                batch_size=1
            )
        except Exception as e:
            raise e
        return retval
