# -*- coding: utf-8 -*-


def normalize_labels(data, screen_width, screen_height):
    # Top Centered webcam

    # Screen
    data['x'] = (data['x']/screen_width*2)-1
    data['y'] = data['y']/screen_height


def normalize_data(data, webcam_width, webcam_height):
    # Top Centered webcam

    # Imgs
    cols_x = ['eye_right_x','eye_left_x','face_x']  # Range [-1,1]
    for col in cols_x:
        data[col] = data[col]/webcam_width*2-1

    cols_width = ['eye_right_width','eye_left_width','face_width']  # Range [0,2]
    for col in cols_width:
        data[col] = data[col]/webcam_width*2

    cols_y = ['eye_right_y','eye_left_y','face_y']  # Range [-1,1]
    for col in cols_y:
        data[col] = (data[col]+webcam_height)/webcam_height-1

    cols_height = ['eye_right_height','eye_left_height','face_height'] # Range [0,2]
    for col in cols_height:
        data[col] = data[col]/webcam_height
