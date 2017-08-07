# -*- coding: utf-8 -*-


import skimage
import skimage.transform


# Bounding boxes by default might be too small.
# This parameter scales the bounding box.
EYE_BBOX_SCALE_WIDTH = 0.9
EYE_BBOX_SCALE_HEIGHT = 1.7


# Features with predictive power
jawline = list(range(1,18))
nose = list(range(29,35))
eyes =[37,40,43,46]
facepoints = jawline + nose + eyes
FEATURES = ['face.x', 'face.y', 'face.width', 'face.height']+[str(i)+'.x' for i in facepoints]+[str(i)+'.y' for i in facepoints]

# FEATURES = ['face.x', 'face.y', 'face.width', 'face.height']+[str(i)+'.x' for i in range(68)]+[str(i)+'.y' for i in range(68)]
TARGETS = ['x','y']

def extract_eye(img, bbox_x, bbox_y, bbox_w, bbox_h, img_size):
    """
    img_width, img_height: int, pixels of the final image.
    """
    try:
        img_eye = img[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
        img_eye = skimage.color.rgb2gray(img_eye)
        img_eye = skimage.exposure.equalize_hist(img_eye)
        img_width, img_height = img_size
        img_eye = skimage.transform.resize(img_eye,(img_height, img_width))  # Rows x Cols
        return img_eye
    except Exception as e:
        print(locals())
        print(e.message)


def dlib2features(features):
    retval = features
    for eye,points in (
        ('eye_left',(36,37,38,39,40,41)),
        ('eye_right',(42,43,44,45,46,47))
    ):
        eye_points = [
            (features[str(p)+'.x'], features[str(p)+'.y'])
            for p in points
        ]
        x = min([i for i,_ in eye_points])
        y = min([i for _,i in eye_points])
        w = max([i for i,_ in eye_points])-x
        h = max([i for _,i in eye_points])-y
        # Scale eye bounding boxes
        w_pad = w*EYE_BBOX_SCALE_WIDTH
        retval[eye+'_x'] = int(x-w_pad//2)
        retval[eye+'_width'] = int(w+w_pad)
        h_pad = h*EYE_BBOX_SCALE_HEIGHT
        retval[eye+'_y'] = int(y-h_pad//2)
        retval[eye+'_height'] = int(h+h_pad)
    return retval
