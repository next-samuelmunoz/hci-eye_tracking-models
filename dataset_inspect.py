# -*- coding: utf-8 -*-
"""Tool to inspect the raw dataset.
"""

import csv

import pygame
from pygame.locals import *
from skimage import io

import config
from utils.data import Data
# from utils.predictor import Predictor
from utils.keras.predictor import Predictor
from utils.features02_dlib import dlib2features


def get_img_id(game_id, timestamp):
    return '{}_{}'.format(game_id, timestamp)

predictor = Predictor(
    path_dlib_model=config.PATH_DLIB,
    model_name='f02_03-07',
    screen_width=config.SCREEN_WIDTH,
    screen_height=config.SCREEN_HEIGHT,
    webcam_width=config.WEBCAM_WIDTH,
    webcam_height=config.WEBCAM_HEIGHT,
    threshold_face_width=config.THRESHOLD_FACE_WIDTH
)

DATASETS_01 = {
    False: None,
    config.PATH_DATA_FEATURES01_COGNITIVE_CSV: None,
    config.PATH_DATA_FEATURES01_DLIB_CSV: None
}


def draw_features01(img, features):
    pygame.draw.rect( # Face
        img, (0,0,255),
        [int(features[x]) for x in ('face_x','face_y','face_width','face_height')],
        3
    )
    pygame.draw.rect( # Left eye
        img, (0,255,255),
        [int(float(features[x])) for x in ('eye_left_x','eye_left_y','eye_left_width','eye_left_height')],
        1
    )
    pygame.draw.rect(  # Right Eye
        img, (0,255,0),
        [int(float(features[x])) for x in ('eye_right_x','eye_right_y','eye_right_width','eye_right_height')],
        1
    )


def loop(data_list):
    pygame.init()
    i_data = 500
    exit = 0
    flag_dot = False
    flag_landmarks = False
    flag_predictor = False
    ds01_key_index = 0
    # Calculate webcam image position
    img = pygame.image.load(data_list[i_data]['img_path'])
    img_w, img_h = img.get_rect().size
    if data_list[i_data]['camera_position'] == 'TC':
        img_pos = ((data_list[i_data]['screen_width']-img_w)/2,0)
    else:
        img_pos = (0,0)

    screen = pygame.display.set_mode(
        (data_list[i_data]['screen_width'],data_list[i_data]['screen_height'])
    )
    while not exit:
        img = pygame.image.load(data_list[i_data]['img_path'])
        print("\n\n-> IMAGE DATA:")
        print(data_list[i_data])
        #Print user screen limits
        ds01_key = list(DATASETS_01.keys())[ds01_key_index]
        print("-> DATASET: {}".format(ds01_key))
        if ds01_key:  # Show detected features
            try:
                if DATASETS_01[ds01_key]==None: # Load dataset
                    DATASETS_01[ds01_key] = {}
                    with open(ds01_key,'r') as fd:
                        csv_reader = csv.DictReader(fd)
                        for row in csv_reader:
                            DATASETS_01[ds01_key][get_img_id(row['game_id'],row['timestamp'])] = row
                features = DATASETS_01[ds01_key][get_img_id(data_list[i_data]['game_id'],data_list[i_data]['timestamp'])]
                draw_features01(img, features)
                print("-> FEATURES:")
                print(features)
            except Exception as e:
                print("[WARNING] Exception: {}".format(e))
        if flag_landmarks:  # Compute landmarks and show
            try:
                f = predictor.dlib_model.extract_features(io.imread(data_list[i_data]['img_path']), -1)
                landmarks = dlib2features(f)
                draw_features01(img, landmarks)
            except Exception as e:
                print("[WARNING] Exception: {}".format(e))
        screen.fill((0,0,0))
        screen.blit(img, img_pos)
        if flag_dot:  # Show dot
            print("-> LOOKING AT: {},{}".format(data_list[i_data]['x'], data_list[i_data]['y']))
            pygame.draw.circle(screen, (255,0,0), (-data_list[i_data]['x']+data_list[i_data]['screen_width'],data_list[i_data]['y']), 25, 0)
        if flag_predictor: # show prediction
            try:
                print("fails?")
                x,y = predictor.predict(io.imread(data_list[i_data]['img_path']))
                print("-> PREDICTION: {},{}".format(int(x),int(y)))
                pygame.draw.circle(screen, (255,255,0), (-int(x)+config.SCREEN_WIDTH,int(y)), 40, 0)
            except Exception as e:
                print("[WARNING] Exception:{}".format(e))
        pygame.display.update()
        click = False
        while not click:
            # for event in pygame.event.get():
            event = pygame.event.wait()
            if event.type == KEYUP:
                if event.key == K_ESCAPE:
                    exit = True
                    click = True
                elif event.key == K_RIGHT:
                    if i_data<len(data_list)-1:
                        i_data += 1
                        click = True
                elif event.key == K_LEFT:
                    if i_data>0:
                        i_data -= 1
                        click = True
                elif event.key == K_d: # Switch dot (where the user looks)
                    flag_dot = False if flag_dot else True
                    click = True
                elif event.key == K_l: # Switch landmarks
                    flag_landmarks = False if flag_landmarks else True
                    click = True
                elif event.key == K_p: # Switch landmarks
                    flag_predictor = False if flag_predictor else True
                    click = True
                elif event.key == K_1: # Switch DS01 mscognitive, dlib
                    ds01_key_index = (ds01_key_index+1) % len(DATASETS_01)
                    click = True
            pygame.event.clear()
    pygame.display.quit()



data = Data(config.PATH_DATA_RAW)
data_list = list(data.iterate())
if data_list:
    print("NUMBER OF SAMPLES: {}".format(len(data_list)))
    loop(data_list)
else:
    print("No data")
