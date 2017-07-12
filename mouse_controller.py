

import time

import numpy as np
from PIL import Image
import pyautogui
import pygame
from pygame.locals import *


import config
from utils.predictor import Predictor
from utils.webcam_pyv4l2Camera import Webcam


DEBUG = False



if __name__=="__main__":
    import os
    os.nice(10)
    predictor = Predictor(
        path_dlib_model=config.PATH_DLIB,
        # model_name='CRD-02',
        model_name='baseline-02',
        # model_name='cnn_simple-09',
        # model_name='cnn-04',
        # model_name='cnn_maxpooling-02',
        screen_width=config.SCREEN_WIDTH,
        screen_height=config.SCREEN_HEIGHT,
        webcam_width=config.WEBCAM_WIDTH,
        webcam_height=config.WEBCAM_HEIGHT,
        threshold_face_width=config.THRESHOLD_FACE_WIDTH
    )
    webcam = Webcam(
        config.WEBCAM_DEVICE,
        config.WEBCAM_WIDTH,
        config.WEBCAM_HEIGHT
    )
    if DEBUG:
        pygame.init()
        screen = pygame.display.set_mode(
            (320, 240)
        )
    flag_exit = False
    while not flag_exit:
        try:
            tstamp = time.time()
            print("Get img")
            img = webcam.get_img()
            print("img!")
            if DEBUG:
                s = pygame.transform.scale(
                    pygame.image.fromstring(img.tobytes(), img.size, img.mode),
                    (320,240)
                )
                screen.blit(s,(0,0))
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == QUIT:
                        flag_exit = True
            # Predict postion
            print("predict")
            x,y = predictor.predict(
                np.asarray(img.convert('L')).copy()  # To grayscale
            )
            print(x,y)
            # Move mouse
            pyautogui.moveTo(x, y, duration=0.1)
            print("end loop {}sg".format(time.time()-tstamp))
        except Exception as e:
            print(e)
    webcam.close()
    if DEBUG:
        pygame.display.quit()
