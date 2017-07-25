# -*- coding: utf-8 -*-
"""Eye tracker predictor
"""

from tensorflow.contrib import keras
import numpy as np

import config
from utils.features_dlib import FeaturesDlib
from utils.features01_dlib import dlib2features01, extract_eye, FEATURES
from utils.normalize import normalize_data

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
            custom_objects={ "euclidean":losses.euclidean }
        )
        # Constants
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.WEBCAM_WIDTH = webcam_width
        self.WEBCAM_HEIGHT = webcam_height
        self.THRESHOLD_FACE_WIDTH = threshold_face_width


    def predict(self, img):
        try:
            # Detect Landmarks
            f = self.dlib_model.extract_features(img, self.THRESHOLD_FACE_WIDTH)
            landmarks = dlib2features01(f)
            # Generate eye arrays
            eyes = {}
            for eye in ('eye_left','eye_right'):
                eyes[eye] = extract_eye(
                    img,
                    landmarks[eye+'_x'], landmarks[eye+'_y'],
                    landmarks[eye+'_width'] ,landmarks[eye+'_height']
                )
            # Normalize data -> [-1,1]
            normalize_data(landmarks, self.WEBCAM_WIDTH, self.WEBCAM_HEIGHT)
            # Make prediction
            img_left = eyes['eye_left']
            img_right = eyes['eye_right']
            x,y = self.model.predict(
                x={
                    'left_imgs': img_left.reshape(1, *img_left.shape),
                    'right_imgs': img_right.reshape(1, *img_right.shape),
                    'features': np.array([[landmarks[x] for x in FEATURES]])
                })[0]
        except Exception as e:
            raise e
        return(  # Rescale to screen coordinates
            (x+1)/2*self.SCREEN_WIDTH,
            y*self.SCREEN_HEIGHT
        )
