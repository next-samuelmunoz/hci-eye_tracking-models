# -*- coding: utf-8 -*-
"""Eye tracker predictor
"""

import config
from .features_dlib import FeaturesDlib
from .features01_dlib import dlib2features01, extract_eye
from .model import Model
from .normalize import normalize_data


class Predictor:

    def __init__(self,
        path_dlib_model, model_name,
        screen_width, screen_height,
        webcam_width, webcam_height,
        threshold_face_width
    ):
        self.dlib_model = FeaturesDlib(path_dlib_model)  # Face landmarks
        self.model = Model(model_name, saved_model=model_name+".final")  # Pretrained model
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
            x,y = self.model.predict_record(landmarks, eyes['eye_left'], eyes['eye_right'])
            # Rescale to screen coordinates
        except Exception as e:
            raise e
        return(
            (x+1)/2*self.SCREEN_WIDTH,
            y*self.SCREEN_HEIGHT
        )
