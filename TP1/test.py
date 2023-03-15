#%%

import numpy as np
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import dct, idct


def padding(img, nl, nc):

    npl = 32-nl % 32
    npc = 32-nc % 32

    ll = img[nl - 1, :][np.newaxis, :]
    repl = ll.repeat(npl, axis=0)
    imgp = np.vstack([img, repl])

    lc = imgp[:, nc-1][:, np.newaxis]
    repc = lc.repeat(npc, axis=1)
    imgp = np.hstack([imgp, repc])

    return imgp

def RGB2YCbCr(R,G,B):
    
    Y = 0.299*R



def encoder(img):

    cmRed = clr.LinearSegmentedColormap.from_list(
        'red', [(0, 0, 0), (1, 0, 0)], 256)
    cmGreen = clr.LinearSegmentedColormap.from_list(
        'green', [(0, 0, 0), (0, 1, 0)], 256)
    cmBlue = clr.LinearSegmentedColormap.from_list(
        'blue', [(0, 0, 0), (0, 0, 1)], 256)
    
    nl = img.shape[0]
    nc = img.shape[1]
    
    img = padding(img, nl, nc)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

