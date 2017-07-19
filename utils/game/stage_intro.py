# -*- coding: utf-8 -*-
"""Game loop
"""

import random

import pygame
from pygame.locals import *

class StageIntro(object):

    def __init__(self, screen, config):
        self.screen = screen
        self.config = config
        self.font = pygame.font.SysFont("monospace", 60)


    def loop(self):
        user_wears_glasses = None
        exit = False
        self._print_screen()
        pygame.display.update()
        while not exit:
            for event in pygame.event.get():
                if event.type == KEYUP:
                    if event.key == K_y:
                        user_wears_glasses = True
                        exit = True
                    elif event.key == K_n:
                        user_wears_glasses = False
                        exit = True
        return user_wears_glasses


    def _print_screen(self):
        self.screen.blit(
            self.font.render("Do you wear glasses? y/n", 1, (255,255,0)),
            (self.config.SCREEN_WIDTH/4, self.config.SCREEN_HEIGHT/2-60)
        )
