# -*- coding: utf-8 -*-
"""Game loop
"""


import pygame
from pygame.locals import *


class StageWebcam(object):

    def __init__(self, screen, webcam, config):
        self.screen = screen
        self.webcam = webcam
        self.config = config
        self.font = pygame.font.SysFont("monospace", 30)


    def loop(self):
        exit = False
        while not exit:
            self._print_screen(
                (pygame.time.get_ticks()/750)%2==1
            )
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == KEYUP:
                    exit = True
                elif event.type == MOUSEBUTTONUP:
                    exit = True

    def _print_screen(self, blink):
        color1 = (255,255,255) if blink else (150,150,150)
        color2 = (255,255,0) if blink else (0,255,0)
        img = self.webcam.get_img()
        self.screen.blit(
            pygame.transform.scale(
                pygame.image.fromstring(img.tobytes(), img.size, img.mode),
                (self.config.SCREEN_WIDTH, self.config.SCREEN_HEIGHT)
            ),
            (0,0)
        )
        y_pos = 15
        pygame.draw.line(self.screen, color2, (self.config.SCREEN_WIDTH/2,0),(self.config.SCREEN_WIDTH/2,self.config.SCREEN_HEIGHT),1)
        pygame.draw.line(self.screen, color2, (0,self.config.SCREEN_HEIGHT/2),(self.config.SCREEN_WIDTH,self.config.SCREEN_HEIGHT/2),1)
        pygame.draw.rect(self.screen, (0,0,0), (0,y_pos,self.config.SCREEN_WIDTH,35), )
        self.screen.blit(
            self.font.render("Please, center yourself in the image and press any key or mouse.", 1, color1),
            (self.config.SCREEN_WIDTH/5, y_pos)
        )
