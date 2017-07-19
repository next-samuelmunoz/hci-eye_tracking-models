# -*- coding: utf-8 -*-
"""Game loop
"""

import time
import random

import pygame
from pygame.locals import *

class StageGame(object):

    def __init__(self, screen, webcam, data, config):
        self.screen = screen
        self.webcam = webcam
        self.data = data
        self.config = config
        self.background = pygame.image.load(self.config.GAME_IMG_BACKGROUND)
        self.target = pygame.image.load(self.config.GAME_IMG_TARGET)
        self.font = pygame.font.SysFont("monospace", 30)


    def loop(self):
        scores = []
        times = []
        fails = self.config.GAME_FAILS
        exit = False
        start_time = pygame.time.get_ticks()
        while not exit:
            click = False
            pygame.mouse.set_pos(
                random.randint(0, self.config.SCREEN_WIDTH),
                random.randint(0, self.config.SCREEN_HEIGHT)
            )
            drift_x, drift_y = random.normalvariate(0,3), random.normalvariate(0,1)
            x=random.randint(0, self.config.SCREEN_WIDTH)
            y=random.randint(0, self.config.SCREEN_HEIGHT)
            self._print_target(x, y, 150, 150)
            while not click and not exit:
                remaining_time = self.config.GAME_TIME - int((pygame.time.get_ticks()-start_time)/1000)
                self._print_stats(sum(scores), remaining_time)
                for event in pygame.event.get():
                    if event.type == KEYUP:
                        if event.key == K_ESCAPE:
                            exit = True
                    elif event.type == MOUSEBUTTONDOWN:
                        mouse_x, mouse_y = event.pos
                        distance = (((x-mouse_x)**2+(y-mouse_y)**2)**0.5)/self.config.GAME_RADIUS
                        if distance<=1:  # Target is hit, take picture!
                            score = int(((1-distance)*6)+5)
                            pygame.time.wait(150)
                            self.webcam.capture(self.data.create_datum(mouse_x, mouse_y, score))
                            pygame.time.wait(350)
                            scores.append(score)
                            click = True
                        else:
                            fails -= 1
                            if fails == 0:
                                exit = True
                if pygame.time.get_ticks()%15==1:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    pygame.mouse.set_pos(
                        mouse_x+drift_x,
                        mouse_y+drift_y
                    )
                if remaining_time<=0:
                    exit = True
        return scores



    def _print_target(self, x, y, size_x, size_y):
        """Print the background and a target.
        x, y: int
            Position of the center of the target.
        x_size, y_size: int
            Size of the target.
        """
        self.screen.blit(self.background, (0,0))  # Background
        new_target = pygame.transform.scale(self.target, (size_x, size_y))
        self.screen.blit(new_target, (x-(size_x/2),y-(size_y/2)) )
        pygame.display.update()


    def _print_stats(self, score, time):
        pygame.draw.rect(self.screen, (0,0,0), (
            self.config.SCREEN_WIDTH-250, 5,
            230, 100
        ), 0)
        pygame.draw.rect(self.screen, (255,255,255), (
            self.config.SCREEN_WIDTH-250, 5,
            228, 100
        ), 1)
        self.screen.blit(
            self.font.render("SCORE: {}".format(score), 1, (255,255,0)),
            (self.config.SCREEN_WIDTH-240, 20)
        )
        self.screen.blit(
            self.font.render("TIME: {}".format(time), 1, (255,255,0)),
            (self.config.SCREEN_WIDTH-240, 70)
        )
        pygame.display.update()
