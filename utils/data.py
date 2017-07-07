# -*- coding: utf-8 -*-
"""Utilities to deal with generated pictures.
"""


import os
import time
import uuid


class Data(object):

    def __init__(self, raw_data_path,
        screen_width='', screen_height='', screen_diagonal='',
        camera_position='', glasses=False
    ):
        """Constructor
        """
        self.raw_data_path = raw_data_path
        glasses_str = 'glasses-yes' if glasses else 'glasses-no'
        self.config_string = "{}_{}_{}_{}_{}". format(
            screen_width, screen_height, screen_diagonal, camera_position,
            glasses_str
        )
        self.game_id = None
        self.game_path = None


    def new_game(self):
        """Generate a new folder this game data.
        """
        self.game_id = uuid.uuid4()
        dir_name = "{}_{}".format(self.game_id, self.config_string)
        self.game_path = os.path.join(self.raw_data_path, dir_name)
        os.makedirs(self.game_path)


    def create_datum(self, x, y, score):
        """Generate a suitable path for the webcam image.
        """
        return os.path.join(
            self.game_path,
            "{epoch}_{x}_{y}_{score}.jpg".format(
                epoch=int(time.time()),
                x=x,
                y=y,
                score=score
            )
        )


    def iterate(self):
        """Iterate over the generated pictures.

        TODO return iterator (picture path, x, y, time, screen params, cam params)
        """
        for directory,subdirs,files in os.walk(self.raw_data_path):
            if files:
                constants = dict(zip(
                    ['game_id','screen_width','screen_height','screen_diagonal','camera_position','glasses'],
                    os.path.basename(directory).split('_')
                ))
                if constants['glasses'] == 'glasses-yes':
                    constants['glasses'] = True
                else:
                    constants['glasses'] = False
                for f in files:
                    retval = dict(zip(
                        ['timestamp', 'x', 'y', 'score'],
                        os.path.splitext(f)[0].split('_')
                    ))
                    retval['img_path'] = os.path.join(directory, f)
                    retval.update(constants)
                    for k in ['timestamp', 'x', 'y', 'score','screen_width','screen_height','screen_diagonal']:
                        retval[k] = int(retval[k])
                    yield retval
