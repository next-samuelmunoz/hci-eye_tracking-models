# -*- coding: utf-8 -*-
""" Just move the mouse.
"""

import numpy as np
import pyautogui


class MoveTo():

    def __init__(self, threshold_radius, duration=0.2):
        self.threshold_radius =  threshold_radius
        self.duration = duration
        self.mouse_pos = np.array([0,0])


    def action(self, mouse_pos):
        if np.linalg.norm(mouse_pos-self.mouse_pos)>self.threshold_radius:
            x,y = mouse_pos
            pyautogui.moveTo(x, y, duration=self.duration)
            self.mouse_pos = mouse_pos
