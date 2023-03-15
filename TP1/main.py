# %%
import numpy as np
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import cv2
from scipy import fftpack as fft


def subsampling(Cb, Cr, ratio):
    CbRatio = ratio[1]/ratio[0]

    if ratio[2] == 0:
        if ratio[1] == 4:
            CrRatio = 0.5
        else:
            CrRatio = CbRatio
    else:
        CrRatio = 1

    dsCbInterp = cv2.resize(Cb, None, fx=CbRatio,
                            fy=CrRatio, interpolation=cv2.INTER_LINEAR)
    dsCrInterp = cv2.resize(Cr, None, fx=CbRatio,
                            fy=CrRatio, interpolation=cv2.INTER_LINEAR)

    plt.figure()
    plt.title("SubSampCb")
    plt.imshow(dsCbInterp, cmap='gray')

    plt.figure()
    plt.title("SubSampCr")
    plt.imshow(dsCrInterp, cmap='gray')

    return dsCbInterp, dsCrInterp


def padding1(img,nl,nc):

    npl = 32-nl%32
    npc = 32-nc%32

    ll = img[nl - 1, :][np.newaxis,:]
    repl = ll.repeat(npl,axis= 0)
    imgp = np.vstack([img,repl])

    lc = imgp[:,nc-1][:,np.newaxis]
    repc = lc.repeat(npc,axis=1)
    imgp = np.hstack([imgp,repc])

    return imgp

def unpadding(img,nl,nc):

    res = img[0:nl:,0:nc:,::]

    return res


def encode(img, nl, nc):

    cmRed = clr.LinearSegmentedColormap.from_list(
        'red', [(0, 0, 0), (1, 0, 0)], 256)
    cmGreen = clr.LinearSegmentedColormap.from_list(
        'green', [(0, 0, 0), (0, 1, 0)], 256)
    cmBlue = clr.LinearSegmentedColormap.from_list(
        'blue', [(0, 0, 0), (0, 0, 1)], 256)
    
    img = padding1(img,nl,nc)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    imgY, imgCb, imgCr = rgb2ycbcr(R, G, B)

    plt.figure()
    plt.title("R")
    plt.imshow(R, cmRed)

    plt.figure()
    plt.title("G")
    plt.imshow(G, cmGreen)

    plt.figure()
    plt.title("B")
    plt.imshow(B, cmBlue)

    plt.figure()
    plt.title("Y")
    plt.imshow(imgY, cmap='gray')

    plt.figure()
    plt.title("Cb")
    plt.imshow(imgCb, cmap='gray')

    plt.figure()
    plt.title("Cr")
    plt.imshow(imgCr, cmap='gray')

    SubCb, SubCr = subsampling(imgCb, imgCr, (4, 2, 2))
    imgDCT(imgY, SubCb, SubCr)
    blockY, blockCb, blockCr = imgBlockDct(imgY, SubCb, SubCr, 8)
    qy, qcb, qcr = quantizer((blockY, blockCb, blockCr), 10)

    qy = DPCM(qy, "y")
    qcb = DPCM(qcb, "cb")
    qcr = DPCM(qcr, "cr")

    return qy, qcb, qcr


def rgb2ycbcr(R,G,B):

    T = np.array([[0.299,0.587,0.114],[-0.168736,-0.331264,0.5],[0.5,-0.418688,-0.081312]])

    Y = T[0,0] * R + T[0,1] * G + T[0,2] * B
    Cb = (T[1,0] * R + T[1,1] * G + T[1,2] * B)+128
    Cr = (T[2,0] * R + T[2,1] * G + T[2,2] * B)+128

    return Y, Cb, Cr


def ycbcr2rgb(Y, Cb, Cr):

    T = np.array([[1,0,1.402], [1,-0.344136, -0.714136],[1,1.722,0]])

    R = Y + T[0,2] * (Cr-128)
    G = Y + T[1,1] * (Cb-128)   + T[1,2] * (Cr-128)
    B = Y + T[2,1] * (Cb-128)   + T[2,2] * (Cr-128)

    R[R>255] = 255
    G[G>255] = 255
    B[B>255] = 255

    R[R<0] = 0
    G[G<0] = 0
    B[B<0] = 0

    return R, G, B
    
# 7
 
def imgDCT(Y, Cb, Cr):
    Ydct = DCT(Y)
    Cbdct = DCT(Cb)
    Crdct = DCT(Cr)

    Ydct = np.log(np.abs(Ydct) + 0.0001)
    Cbdct = np.log(np.abs(Cbdct) + 0.0001)
    Crdct = np.log(np.abs(Crdct) + 0.0001)

    plt.figure()
    plt.title("DCT Y")
    plt.imshow(Ydct, cmap='gray')

    plt.figure()
    plt.title("DCT Cb")
    plt.imshow(Cbdct, cmap='gray')

    plt.figure()
    plt.title("DCT Cr")
    plt.imshow(Crdct, cmap='gray')

    return Ydct, Cbdct, Crdct


# DCT com blocos

def imgBlockDct(y, cb, cr, block):
    Ydct = BlockDct(y, size=block)
    Cbdct = BlockDct(cb, size=block)
    Crdct = BlockDct(cr, size=block)

    Yblock = np.log(np.abs(Ydct) + 0.0001)
    Cbblock = np.log(np.abs(Cbdct) + 0.0001)
    Crblock = np.log(np.abs(Crdct) + 0.0001)

    plt.figure()
    plt.title("DCT Block Y")
    plt.imshow(Yblock, cmap='gray')

    plt.figure()
    plt.title("DCT Block Cb")
    plt.imshow(Cbblock, cmap='gray')

    plt.figure()
    plt.title("DCT Block Cr")
    plt.imshow(Crblock, cmap='gray')

    return Ydct, Cbdct, Crdct


def DCT(x):
    return fft.dct(fft.dct(x, norm="ortho").T, norm="ortho").T


def idct(x):
    return fft.idct(fft.idct(x, norm="ortho").T, norm="ortho").T


def BlockDct(x, size):
    h, w = x.shape
    newImg = np.zeros(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = DCT(x[i:i+size, j:j+size])
    return newImg

# 8

QY = np.array([[16,  11,  10,  16,  24,  40,  51,  61],
               [12,  12,  14,  19,  26,  58,  60,  55],
               [14,  13,  16,  24,  40,  57,  69,  56],
               [14,  17,  22,  29,  51,  87,  80,  62],
               [18,  22,  37,  56,  68, 109, 103,  77],
               [24,  35,  55,  64,  81, 104, 113,  92],
               [49,  64,  78,  87, 103, 121, 120, 101],
               [72,  92,  95,  98, 112, 100, 103,  99]])


QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
               [18, 21, 26, 66, 99, 99, 99, 99],
               [24, 26, 56, 99, 99, 99, 99, 99],
               [47, 66, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99]])


def quantizer(ycbcr, quality):
    y, cb, cr = ycbcr
    revQ = (100 - quality) / 50 if quality >= 50 else 50 / quality
    QsY = np.round(QY * revQ)
    QsC = np.round(QC * revQ)

    QsY[QsY > 255] = 255
    QsC[QsC > 255] = 255
    QsY[QsY < 1] = 1
    QsC[QsC < 1] = 1 
    QsC = QsC.astype(np.uint8)
    QsY = QsY.astype(np.uint8)
    

    qy = np.empty(y.shape, dtype=y.dtype)
    qcb = np.empty(cb.shape, dtype=cb.dtype)
    qcr = np.empty(cr.shape, dtype=cr.dtype)

    for i in range(0, y.shape[0], 8):
        for j in range(0, y.shape[1], 8):
            qy[i:i+8, j:j+8] = y[i:i+8, j:j+8] / QsY
    qy = np.round(qy).astype(np.int16)

    for i in range(0, cb.shape[0], 8):
        for j in range(0, cb.shape[1], 8):
            qcb[i:i+8, j:j+8] = cb[i:i+8, j:j+8] / QsC
    qcb = np.round(qcb).astype(np.int16)

    for i in range(0, cr.shape[0], 8):
        for j in range(0, cr.shape[1], 8):
            qcr[i:i+8, j:j+8] = cr[i:i+8, j:j+8] / QsC
    qcr = np.round(qcr).astype(np.int16)

    ly = np.log(np.abs(qy) + 0.0001)
    lcb = np.log(np.abs(qcb) + 0.0001)
    lcr = np.log(np.abs(qcr) + 0.0001)

    plt.figure()
    plt.title("Quantized Y")
    plt.imshow(ly, cmap='gray')

    plt.figure()
    plt.title("Quantized Cb")
    plt.imshow(lcb, cmap='gray')

    plt.figure()
    plt.title("Quantized Cr")
    plt.imshow(lcr, cmap='gray')

    return qy, qcb, qcr

# 9
    
def DPCM(imgDCT_Q, channel):

    first = True
    imgDPCM = imgDCT_Q.copy()
    dc0 = imgDPCM[0, 0]
    nl, nc = imgDPCM.shape
    for i in range(0, nl, 8):
        for j in range(0, nc, 8):
            if i == 0 and j == 0:
                continue
            dc = imgDCT_Q[i, j]
            diff = dc - dc0
            # print("yDPCM: " + str(diff))
            dc0 = dc
            imgDPCM[i, j] = diff

    plt.figure()
    plt.title('DPCM (' + channel + ')')
    plt.imshow(np.log(np.abs(imgDPCM) + 0.0001), "gray")

    return imgDPCM

def decoder(Y,Cb,Cr):

    R, G, B = ycbcr2rgb(Y,Cb,Cr)

    res = unpadding(img,nl,nc)

    return res





if __name__ == "__main__":
    img_bgr = cv2.imread('./images/peppers.bmp')
    print(img_bgr.shape)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.title("Original")
    plt.imshow(img)
    nl = img.shape[0]
    nc = img.shape[1]
    img = encode(img,nl,nc)


# %%
