

from time import sleep

import numpy as np
from PIL import Image
import pyautogui
import pygame
from pygame.locals import *


import config
from utils.predictor import Predictor
from utils.webcam_pyv4l2Camera import Webcam
# from PyV4L2Camera.camera import Camera




if __name__=="__main__":
    import os
    os.nice(10)
    screen = pygame.display.set_mode(
        (320, 240)
    )

    webcam = Webcam(
        config.WEBCAM_DEVICE,
        config.WEBCAM_WIDTH,
        config.WEBCAM_HEIGHT
    )
    predictor = Predictor(
        path_dlib_model=config.PATH_DLIB,
        model_name='CRD-02',
        screen_width=config.SCREEN_WIDTH,
        screen_height=config.SCREEN_HEIGHT,
        webcam_width=config.WEBCAM_WIDTH,
        webcam_height=config.WEBCAM_HEIGHT,
        threshold_face_width=config.THRESHOLD_FACE_WIDTH
    )

    pygame.init()
    flag_exit = False
    while not flag_exit:
        try:
            print("Get img")
            img = webcam.get_img()
            print("img!")
            # Print Image
            s = pygame.transform.scale(
                pygame.image.fromstring(img.tobytes(), img.size, img.mode),
                (320,240)
            )
            screen.blit(s,(0,0))
            pygame.display.update()
            # Predict postion
            for event in pygame.event.get():
                if event.type == QUIT:
                    flag_exit = True

            print("predict")
            x,y = predictor.predict(
                np.asarray(img).copy()
            )
            print(x,y)
            # Move mouse
            pyautogui.moveTo(x, y, duration=0.25)
        except Exception as e:
            print(e)
    webcam.close()
    pygame.display.quit()
