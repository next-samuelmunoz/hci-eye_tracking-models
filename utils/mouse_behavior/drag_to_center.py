# -*- coding: utf-8 -*-
""" Mouse behavior for the mouse controller.
"""


import time

import numpy as np

import pyautogui

class Drag2Center():

    def __init__(self, threshold_radius, radius, screen_width, screen_height, fire_sg=2, duration=0.2 ):
        """
        fire: fire in seconds to remain quiet before firing the drag event
        """
        self.threshold_radius = threshold_radius
        self.radius = radius
        self.screen_width = screen_width
        self.screen_height  = screen_height
        self.fire_sg = fire_sg
        self.duration = duration
        self.time_moved = False
        self.center_pos = np.array([self.screen_width, self.screen_height])/2
        self.action(self.center_pos)  # move mouse to a safe place
        self.mouse_pos = np.array([0,0])  # Position in a previous iteration


    def action(self, mouse_pos):
        if np.linalg.norm(mouse_pos-self.center_pos) > self.radius:  # Scroll zone
            now = time.time()
            if np.linalg.norm(mouse_pos-self.mouse_pos)>self.threshold_radius:  # Moved, don't scroll
                self.mouse_pos = mouse_pos
                self.time_moved = now
            else:
                if self.time_moved and now-self.time_moved>self.fire_sg:
                    x, y = mouse_pos
                    pyautogui.moveTo(x, y)
                    pyautogui.dragTo(
                        self.screen_width/2,
                        self.screen_height/2,
                        duration = self.duration
                    )
                    self.mouse_pos = self.center_pos
                    self._hide_mouse()
                    self.time_moved = now
        # x, y = mouse_pos
        # pyautogui.moveTo(x, y)  # TODO delete


    def _hide_mouse(self):
        """Put the cursor on a corner so it cannot be seen.
        """
        pyautogui.moveTo(0, self.screen_width)
