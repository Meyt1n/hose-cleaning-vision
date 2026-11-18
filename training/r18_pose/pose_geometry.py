import numpy as np
from PIL import Image
def affine(image, xy, angle, scale, shift, flip=False):
    center = np.array([127.5, 127.5])
    rad = np.deg2rad(angle)
    (c, s) = (np.cos(rad), np.sin(rad))
    rot = np.array([[c, s], [-s, c]]) * scale
    mat = np.eye(3)
    mat[:2, :2] = rot
    mat[:2, 2] = center + np.array(shift) - rot @ center
    inv = np.linalg.inv(mat)
    image = image.transform((256, 256), Image.Transform.AFFINE, tuple(inv[:2].ravel()), resample=Image.Resampling.BILINEAR, fillcolor=(124, 116, 104))
    points = xy @ rot.T + mat[:2, 2]
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        points[:, 0] = 255 - points[:, 0]
    return (image, points)
