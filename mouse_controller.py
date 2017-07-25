
import argparse
import os
import time

import numpy as np
import pygame
from pygame.locals import *


import config
# from utils.predictor import Predictor
from utils.keras.predictor import Predictor
from utils.webcam_pyv4l2Camera import Webcam
import utils.mouse_behavior



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--behavior",
        help="Set the mouse behavior",
        choices=['move', 'drag2center', 'scroll'],
        default="move",
        type=str
    )
    parser.add_argument("-d","--debug", action='store_true')
    args = parser.parse_args()
    # Debug mode
    if args.debug:
        DEBUG = True
    else:
        DEBUG = False
    # Set mouse behavior
    if args.behavior=="drag2center":
        mouse = utils.mouse_behavior.Drag2Center(
            threshold_radius=80,
            radius= config.SCREEN_HEIGHT*0.3,
            screen_width=config.SCREEN_WIDTH,
            screen_height=config.SCREEN_HEIGHT,
            fire_sg=0.5,
            duration=0.7
        )
    elif args.behavior=="scroll":
        mouse = utils.mouse_behavior.Scroll(
            border= int(config.SCREEN_HEIGHT/3),
            screen_height=config.SCREEN_HEIGHT,
            scroll_move=1,
            fire_sg=0.05,
            # duration=1.5
        )
    else:
        mouse = utils.mouse_behavior.MoveTo(
            threshold_radius=80,
            duration=0.2
        )


    os.nice(10)
    predictor = Predictor(
        path_dlib_model=config.PATH_DLIB,
        # model_name='DSR-07',
        # model_name='CRD-02',
        model_name='baseline-08',
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
            img = webcam.get_img()
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
            x,y = predictor.predict(
                np.asarray(img.convert('L')).copy()  # To grayscale
            )
            mouse.action(np.array([x,y]))
            if DEBUG:
                print("Loop {}sg".format(time.time()-tstamp))
        except Exception as e:
            print(e)
    webcam.close()
    if DEBUG:
        pygame.display.quit()
