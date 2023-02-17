
#%%
import numpy as np
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import cv2

img = plt.imread('images/barn_mountains.bmp')

def padding(img,nl,nc):

    npl = 32-nl%32
    npc = 32-nc%32

    ll = img[nl - 1, :][np.newaxis,:]
    repl = ll.repeat(npl,axis= 0)
    imgp = np.vstack([img,repl])

    lc = imgp[:,nc-1][:,np.newaxis]
    repc = lc.repeat(npc,axis=1)
    imgp = np.hstack([imgp,repc])

    return imgp



def encode(img,nl,nc):

    cmRed = clr.LinearSegmentedColormap.from_list('red',[(0,0,0),(1,0,0)],256)
    cmGreen = clr.LinearSegmentedColormap.from_list('green',[(0,0,0),(0,1,0)],256)
    cmBlue = clr.LinearSegmentedColormap.from_list('blue',[(0,0,0),(0,0,1)],256)

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    plt.figure()
    plt.imshow(R,cmRed)
    plt.figure()
    plt.imshow(G,cmGreen)
    plt.figure()
    plt.imshow(B,cmBlue)

    img = padding(img,nl,nc)

    plt.imshow(img)


    
img = plt.imread('images/barn_mountains.bmp')
nl = img.shape[0]
nc = img.shape[1]



encode(img,nl,nc)





#%%
