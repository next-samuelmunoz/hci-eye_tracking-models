# -*- coding: utf-8 -*-
""" Use webcam with PyV4L2Camera.
"""

from time import sleep

from threading import Thread

from PIL import Image
from PyV4L2Camera.camera import Camera


class WebcamThread(Thread):

    def __init__(self, device, width, height, color='RGB'):
        '''Intialize device
        '''
        self._cam = Camera(device, width, height)
        self.width, self.height = self._cam.width, self._cam.height
        self.running = True
        self.img = None
        self.t_wait = 1.0/60  # Webcam operates at 30FPS
        super().__init__()


    def run(self):
        '''Thread loop. Read continuously from cam buffer.
        '''
        while self.running:
            self.img = self._cam.get_frame()
            sleep(self.t_wait)
        self._cam.close()


    def capture(self, path_file):
        '''Capture image into a file
        '''
        image = self.get_img()
        image.save(path_file)


    def get_img(self):
        return Image.frombytes('RGB', (self.width, self.height), self.img, 'raw' , 'RGB')


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
        return self.thread.get_img()


    def close(self):
        self.thread.close()
        self.thread.join()
