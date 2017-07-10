# -*- coding: utf-8 -*-
"""Constants
"""

#
### CONSTANTS
#

# Space where de user is looking at
SCREEN_WIDTH = 1366
SCREEN_HEIGHT = 768


# Webcam
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
WEBCAM_DEVICE = "/dev/video0"
"""
-NOT USED-
Position of the camera respect to the screen.
Format is XX where X can be:
T: top
B: bottom
C: center
L: left
R: right
"""
WEBCAM_POSITION = 'TC'


#
### game01
#

# IMAGES
IMG_BACKGROUND = 'img/game01/bg-sky.jpg'
IMG_TARGET = 'img/game01/target01.png'

# OTHER
TIME_GAME = 60  # Seconds
FAILS = 5  # Hits a user can fail
RADIUS = 75.0  # Radius of the target, centered in the image



#
### FACIAL LANDMARK DETECTOR: Dlib
#

PATH_DLIB = 'data/shape_predictor_68_face_landmarks.dat'



#
### DATASET: raw
#

PATH_DATA_RAW = 'data/raw'



#
### DATASET: features01
#

FEATURES01_EYES_SIZE = (30,20)  # Eye img shape
THRESHOLD_FACE_WIDTH = 200  # Minimum face width


# features01_cognitive
PATH_DATA_FEATURES01_COGNITIVE = 'data/features01_cognitive/'
PATH_DATA_FEATURES01_COGNITIVE_CSV = PATH_DATA_FEATURES01_COGNITIVE+'features.csv'

# features01_dlib
PATH_DATA_FEATURES01_DLIB = "data/features01_dlib/"
PATH_DATA_FEATURES01_DLIB_CSV = PATH_DATA_FEATURES01_DLIB+"features.csv"

# features01_dlib_augmented
PATH_DATA_FEATURES01_DLIB_AUGMENTED = "data/features01_dlib_augmented/"
PATH_DATA_FEATURES01_DLIB_AUGMENTED_CSV = PATH_DATA_FEATURES01_DLIB_AUGMENTED+"features.csv"

# features01_dlib_augmented
PATH_DATA_FEATURES01_DLIB_AUGMENTED_NORM = "data/features01_dlib_augmented_norm/"
PATH_DATA_FEATURES01_DLIB_AUGMENTED_NORM_CSV = "data/data_norm.csv"
PATH_DATA_FEATURES01_DLIB_AUGMENTED_NORM_IMGS_LEFT = "data/img_left_norm"
PATH_DATA_FEATURES01_DLIB_AUGMENTED_NORM_IMGS_RIGHT = "data/img_right_norm"
