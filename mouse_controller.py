

from time import sleep

import pyautogui
import pygame
from pygame.locals import *


import config
from utils.predictor import Predictor
from utils.webcam_pygame import Webcam



if __name__=="__main__":
    webcam = Webcam(
        config.WEBCAM_DEVICE,
        config.WEBCAM_WIDTH,
        config.WEBCAM_HEIGHT
    )
    sleep(3)
    predictor = Predictor(
        path_dlib_model=config.PATH_DLIB,
        model_name='CRD-02',
        screen_width=config.SCREEN_WIDTH,
        screen_height=config.SCREEN_HEIGHT,
        webcam_width=config.WEBCAM_WIDTH,
        webcam_height=config.WEBCAM_HEIGHT,
        threshold_face_width=config.THRESHOLD_FACE_WIDTH
    )

    while True:
        try:
            img = webcam.get_img()
            x,y = predictor.predict(
                pygame.surfarray.array3d(img).swapaxes(0,1)
            )
            pyautogui.moveTo(x, y, duration=0.25)
            print(x,y)
        except Exception as e:
            print(e)
    webcam.close()
