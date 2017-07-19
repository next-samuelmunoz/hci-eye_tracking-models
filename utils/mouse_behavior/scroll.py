# -*- coding: utf-8 -*-
""" Mouse behavior for the mouse controller.
"""


import time

import numpy as np

import pyautogui

class Scroll():

    def __init__(self, border, screen_height, scroll_move, fire_sg=2, duration=0.2 ):
        """
        +-----------------------+
        |     Scroll Up area    |
        +-----------------------+
        |                       |
        |    Middle screen      |
        |                       |
        +-----------------------+
        |    Scroll Down area   |
        +-----------------------+
        """
        self.border = border  # Height of the scroll area
        self.screen_height  = screen_height
        self.scroll_move = scroll_move # Amount of scroll
        self.fire_sg = fire_sg  # Time to keep eyes in scroll area before scrolling
        self.duration = duration  # Scrolling speed
        self.time_in_scroll = False  # First time when eyes in scroll area


    def _in_scroll_area(self, y):
        """Return the scroll area.
        """
        retval = False
        if y<self.border:
            retval = "UP"
        if y>(self.screen_height-self.border):
            retval = "DOWN"
        return retval


    def action(self, mouse_pos):
        _, y = mouse_pos  # Won't need x coordinate
        scroll = self._in_scroll_area(y)
        if scroll:
            now = time.time()
            if self.time_in_scroll:
                if now-self.time_in_scroll>self.fire_sg:
                    direction = 1 if scroll=="UP" else -1
                    pyautogui.scroll(self.scroll_move*direction)
                    self.time_in_scroll = False
            else:
                self.time_in_scroll = time.time()
        else:
            self.time_in_scroll = False
