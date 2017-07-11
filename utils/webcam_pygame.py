# -*- coding: utf-8 -*-
""" Use webcam with pygame.
IMPORTANT:
    I could not get resolutions higher than 640x480.
"""

from threading import Thread

import pygame
from pygame import camera


class WebcamThread(Thread):

    def __init__(self, device, width, height, color='RGB'):
        '''Intialize device
        '''
        camera.init()
        self._cam = camera.Camera(device, (width,height), color)
        self._cam.start()
        self.running = True
        self.img = None
        super().__init__()


    def run(self):
        '''Thread loop. Read continuously from cam buffer.
        '''
        while self.running:
            self.img = self._cam.get_image()
        self._cam.stop()


    def capture(self, path_file):
        '''Capture image into a file
        '''
        pygame.image.save(self.img, path_file)


    def close(self):
        '''Stop webcam and thread
        '''
        self.running = False


class Webcam(object):
    '''Wrapper over the thread.
    '''

    def __init__(self, *args, **kwargs):
        self.thread = WebcamThread(*args, **kwargs)
        self.thread.start()

    def capture(self, *args, **kwargs):
        self.thread.capture(*args, **kwargs)


    def get_img(self):
        return self.thread.img


    def close(self):
        self.thread.close()
        self.thread.join()



if __name__ == "__main__":
    camera.init()
    cams = camera.list_cameras()
    if cams:
        print("Detected webcams:")
        for c in cams:
            print(" {}".format(c))
    else:
        print("No webcams detected.")
