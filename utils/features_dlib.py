# -*- coding: utf-8 -*-
"""
Extract face features with dlib.

pip package: dlib
deb packages: libboost-python-dev, cmake
Based in: http://dlib.net/face_landmark_detection.py.html

Facial shape predictor: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""


import csv
import os
import dlib
from skimage import io
import skimage.color
import skimage.transform


class FeaturesDlib(object):
    def __init__(self, predictor_path, scale_factor=0.5 ):
        self.scale_factor = scale_factor  # downsample image on hog face detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)


    def extract_features(self, img, threshold_face_width):
        retval = None
        img = skimage.color.rgb2gray(img)
        faces = self.detector(img, 0)
        if faces:
            biggest_face = max(faces,key=lambda x:x.area())
            landmarks = self.predictor(img, biggest_face)
            if landmarks:
                if biggest_face.width()>=threshold_face_width:
                    retval = {
                        'face.x': biggest_face.left(),
                        'face.y': biggest_face.top(),
                        'face.width': biggest_face.width(),
                        'face.height': biggest_face.height(),
                    }
                    for i,point in enumerate(landmarks.parts()):
                        retval['{}.x'.format(i)] = point.x
                        retval['{}.y'.format(i)] = point.y
                else:
                    raise Exception("Face width is below threshold: {}".format(threshold_face_width))
            else:
                raise Exception("No landmarks detected")
        else:
            raise Exception("No face detected")
        return retval
