# -*- coding: utf-8 -*-
"""Tool to inspect the raw dataset.
"""

import csv
import os

import pygame
from pygame.locals import *
from skimage import io

import config
from utils.data import Data



def loop(data_list):
    pygame.init()
    i_data = 0
    exit = 0
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
        print("\n\n-> IMAGE DATA: {}\t ERROR: {}".format(i_data, data_list[i_data]['error']))
        # print(data_list[i_data])
        screen.fill((0,0,0))
        screen.blit(img, img_pos)
        # Show dot
        print("-> LOOKING AT: {},{}".format(data_list[i_data]['x'], data_list[i_data]['y']))
        pygame.draw.circle(screen, (255,0,0), (-data_list[i_data]['x']+data_list[i_data]['screen_width'],data_list[i_data]['y']), 25, 0)
        pygame.display.update()
        click = False
        while not click:
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
                elif event.key == K_d: # Delete the sample
                    print("Delete")
                    os.remove(data_list[i_data]['img_path'])
                    i_data += 1
                    click = True
            pygame.event.clear()
    pygame.display.quit()



with open(config.PATH_DATA_RAW_RANKED_ERRORS, 'r') as fd:
    errors = {
        img:error
        for img, error in csv.reader(fd)
    }

data = Data(config.PATH_DATA_RAW)
data_list = list(data.iterate())
for d in data_list:
    d['error'] = errors[d['img_path']]
data_list.sort(
    key=lambda x: x['error'],
    reverse=True
)

if data_list:
    print("NUMBER OF SAMPLES: {}".format(len(data_list)))
    loop(data_list)
else:
    print("No data")
