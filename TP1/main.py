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


def upsampling(dsCb, dsCr, ratio):
    # O reverse dos ratios é feito aqui
    CbRatio = ratio[0]/ratio[1]

    if ratio[2] == 0:
        if ratio[1] == 4:
            CrRatio = 0.5
        else:
            CrRatio = CbRatio
    else:
        CrRatio = 1

    upsCbInterp = cv2.resize(dsCb, None, fx=CbRatio,
                             fy=CrRatio, interpolation=cv2.INTER_LINEAR)
    upsCrInterp = cv2.resize(dsCr, None, fx=CbRatio,
                             fy=CrRatio, interpolation=cv2.INTER_LINEAR)

    # Caso o tamanho não corresponda, recorta-se os canais upsampled para serem do mm tamanho q os originais
    if upsCbInterp.shape != upsCrInterp.shape:
        min_dim = min(upsCbInterp.shape[0], upsCrInterp.shape[1])
        upsCbInterp = upsCbInterp[:min_dim, :min_dim]
        upsCrInterp = upsCrInterp[:min_dim, :min_dim]

    return upsCbInterp, upsCrInterp


def padding1(img, nl, nc):

    npl = 32-nl % 32
    npc = 32-nc % 32

    ll = img[nl - 1, :][np.newaxis, :]
    repl = ll.repeat(npl, axis=0)
    imgp = np.vstack([img, repl])

    lc = imgp[:, nc-1][:, np.newaxis]
    repc = lc.repeat(npc, axis=1)
    imgp = np.hstack([imgp, repc])

    return imgp


def unpadding(img, nl, nc):

    res = img[0:nl:, 0:nc:, ::]

    return res


def encode(img, nl, nc, quality):

    cmRed = clr.LinearSegmentedColormap.from_list(
        'red', [(0, 0, 0), (1, 0, 0)], 256)
    cmGreen = clr.LinearSegmentedColormap.from_list(
        'green', [(0, 0, 0), (0, 1, 0)], 256)
    cmBlue = clr.LinearSegmentedColormap.from_list(
        'blue', [(0, 0, 0), (0, 0, 1)], 256)

    img = padding1(img, nl, nc)

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

    SubCb, SubCr = subsampling(imgCb, imgCr, (4, 2, 0))
    print(SubCb.shape)
    imgDCT(imgY, SubCb, SubCr)
    blockY, blockCb, blockCr = imgBlockDct(imgY, SubCb, SubCr, 8)
    qy, qcb, qcr = quantizer((blockY, blockCb, blockCr), quality)

    qy = DPCM(qy, "y")
    qcb = DPCM(qcb, "cb")
    qcr = DPCM(qcr, "cr")

    return qy, qcb, qcr


def decoder(qy, qcb, qcr, nl, nc, quality):

    qy = iDPCM(qy, "y")
    qcb = iDPCM(qcb, "cb")
    qcr = iDPCM(qcr, "cr")
    y_dct, cb_dct, cr_dct = iQuantizer((qy, qcb, qcr), quality)
    y, cb, cr = idct_block(y_dct, cb_dct, cr_dct, block=8)
    cb, cr = upsampling(cb, cr, (4, 2, 0))

    R, G, B = ycbcr2rgb(y, cb, cr)

    decoded = np.zeros((nl, nc, 3), dtype=np.uint8)

    decoded[:, :, 0] = R[0:nl:, 0:nc:]
    decoded[:, :, 1] = G[0:nl:, 0:nc:]
    decoded[:, :, 2] = B[0:nl:, 0:nc:]

    #res = unpadding(img, nl, nc)

    plt.figure()
    plt.title("Decoded")
    plt.imshow(decoded)

    return decoded


T = np.array([[0.299, 0.587, 0.114], [-0.168736, -
                                      0.331264, 0.5], [0.5, -0.418688, -0.081312]])


def rgb2ycbcr(R, G, B):

    Y = T[0, 0] * R + T[0, 1] * G + T[0, 2] * B
    Cb = (T[1, 0] * R + T[1, 1] * G + T[1, 2] * B)+128
    Cr = (T[2, 0] * R + T[2, 1] * G + T[2, 2] * B)+128

    return Y, Cb, Cr


def ycbcr2rgb(Y, Cb, Cr):

    Ti = np.linalg.inv(T)

    R = Y + Ti[0, 2] * (Cr-128)
    G = Y + Ti[1, 1] * (Cb-128) + Ti[1, 2] * (Cr-128)
    B = Y + Ti[2, 1] * (Cb-128) + Ti[2, 2] * (Cr-128)

    R[R > 255] = 255
    G[G > 255] = 255
    B[B > 255] = 255

    R[R < 0] = 0
    G[G < 0] = 0
    B[B < 0] = 0

    return R.astype(np.uint8), G.astype(np.uint8), B.astype(np.uint8)

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

    # Para mostrar no grafico
    Ybshow = np.log(np.abs(Ydct) + 0.0001)
    Cbshow = np.log(np.abs(Cbdct) + 0.0001)
    Crshow = np.log(np.abs(Crdct) + 0.0001)

    plt.figure()
    plt.title("DCT Block Y")
    plt.imshow(Ybshow, cmap='gray')

    plt.figure()
    plt.title("DCT Block Cb")
    plt.imshow(Cbshow, cmap='gray')

    plt.figure()
    plt.title("DCT Block Cr")
    plt.imshow(Crshow, cmap='gray')

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


def idct_block(y_dct, cb_dct, cr_dct, block):

    y_inv = blockIdct(y_dct, size=block)
    cb_inv = blockIdct(cb_dct, size=block)
    cr_inv = blockIdct(cr_dct, size=block)

    # Mostrar no grafico
    Yshow = np.log(np.abs(y_inv) + 0.0001)
    Cbshow = np.log(np.abs(cb_inv) + 0.0001)
    Crshow = np.log(np.abs(cr_inv) + 0.0001)

    plt.figure()
    plt.title("IDCT Block Y")
    plt.imshow(Yshow, cmap='gray')

    plt.figure()
    plt.title("IDCT Block Cb")
    plt.imshow(Cbshow, cmap='gray')

    plt.figure()
    plt.title("IDCT Block Cr")
    plt.imshow(Crshow, cmap='gray')

    return y_inv, cb_inv, cr_inv


def blockIdct(x: np.ndarray, size):
    h, w = x.shape
    newImg = np.zeros(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = idct(x[i:i+size, j:j+size])
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

    qyshwow = np.log(np.abs(qy) + 0.0001)
    qcbshow = np.log(np.abs(qcb) + 0.0001)
    qcrshow = np.log(np.abs(qcr) + 0.0001)

    plt.figure()
    plt.title("Quantized Y")
    plt.imshow(qyshwow, cmap='gray')

    plt.figure()
    plt.title("Quantized Cb")
    plt.imshow(qcbshow, cmap='gray')

    plt.figure()
    plt.title("Quantized Cr")
    plt.imshow(qcrshow, cmap='gray')

    return qy, qcb, qcr


def iQuantizer(ycbcr: tuple, qf: int):
    qy, qcb, qcr = ycbcr
    sf = (100 - qf) / 50 if qf >= 50 else 50 / qf
    QsY = np.round(QY * sf)
    QsC = np.round(QC * sf)

    QsY[QsY > 255] = 255
    QsC[QsC > 255] = 255
    QsY[QsY < 1] = 1
    QsC[QsC < 1] = 1
    QsC = QsC.astype(np.uint8)
    QsY = QsY.astype(np.uint8)

    y = np.empty(qy.shape, dtype=qy.dtype)
    cb = np.empty(qcb.shape, dtype=qcb.dtype)
    cr = np.empty(qcr.shape, dtype=qcr.dtype)

    for i in range(0, y.shape[0], 8):
        for j in range(0, y.shape[1], 8):
            y[i:i+8, j:j+8] = qy[i:i+8, j:j+8] * QsY

    for i in range(0, cb.shape[0], 8):
        for j in range(0, cb.shape[1], 8):
            cb[i:i+8, j:j+8] = qcb[i:i+8, j:j+8] * QsC

    for i in range(0, cr.shape[0], 8):
        for j in range(0, cr.shape[1], 8):
            cr[i:i+8, j:j+8] = qcr[i:i+8, j:j+8] * QsC

    qyshow = np.log(np.abs(qy) + 0.0001)
    qcbshow = np.log(np.abs(qcb) + 0.0001)
    qcrshow = np.log(np.abs(qcr) + 0.0001)

    plt.figure()
    plt.title("iQuantized Y  Qualidade:" + str(qf))
    plt.imshow(qyshow, cmap='gray')

    plt.figure()
    plt.title("iQuantized (Cb) Qualidade:" + str(qf))
    plt.imshow(qcbshow, cmap='gray')

    plt.figure()
    plt.title("iQuantized Cr Qualidade: " + str(qf))
    plt.imshow(qcrshow, cmap='gray')

    return y.astype(float), cb.astype(float), cr.astype(float)

# 9


def DPCM(imgDCT_Q, channel):

    imgDPCM = imgDCT_Q.copy()
    dc0 = imgDPCM[0, 0]
    nl, nc = imgDPCM.shape
    for i in range(0, nl, 8):
        for j in range(0, nc, 8):
            if i == 0 and j == 0:
                continue
            dc = imgDCT_Q[i, j]
            diff = dc - dc0
            dc0 = dc
            imgDPCM[i, j] = diff

    plt.figure()
    plt.title('DPCM (' + channel + ')')
    plt.imshow(np.log(np.abs(imgDPCM) + 0.0001), "gray")

    return imgDPCM


def iDPCM(imgDCT_Q, channel):

    first = True

    # DPCM 8x8
    imgDPCM = imgDCT_Q.copy()
    dc0 = imgDPCM[0, 0]
    nl, nc = imgDPCM.shape
    for i in range(0, nl, 8):
        for j in range(0, nc, 8):
            if i == 0 and j == 0:
                continue
            dc = imgDCT_Q[i, j]
            s = dc + dc0
            if first and channel == "y":
                first = False
            dc0 = s
            imgDPCM[i, j] = s

    imgshow = np.log(np.abs(imgDPCM) + 0.0001)
    plt.figure()
    plt.title('iDPCM (' + channel + ')')
    plt.imshow(imgshow, "gray")

    return imgDPCM


if __name__ == "__main__":
    img_bgr = cv2.imread('./images/peppers.bmp')
    print(img_bgr.shape)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    yOr, CbOr, CrOr = rgb2ycbcr(img[:, :, 0], img[:, :, 1], img[:, :, 2])
    plt.figure()
    plt.title("Original")
    plt.imshow(img)
    nl = img.shape[0]
    nc = img.shape[1]
    quality = 10
    qy, qcb, qcr = encode(img, nl, nc, quality)
    imgRec = decoder(qy, qcb, qcr, nl, nc, quality,)
    yRec, CbRec, CrRec = rgb2ycbcr(
        imgRec[:, :, 0], imgRec[:, :, 1], imgRec[:, :, 2])
    E = abs(yOr-yRec)
    plt.figure()
    plt.title('E')
    plt.imshow(E, "gray")
    print("E: " + str(np.mean(E)))
    MSE = np.sum((img.astype(float) - imgRec.astype(float))**2)/(nl*nc)
    print("MSE: " + str(MSE))
    RMSE = np.sqrt(MSE)
    P = np.sum((img.astype(float))**2)/(nl*nc)
    SNR = 10 * np.log10(P/MSE)
    PSNR = 10 * np.log10((np.max(img.astype(float))**2)/MSE)
    print("RMSE: " + str(RMSE))
    print("SNR: " + str(SNR))
    print("PSNR: " + str(PSNR))


# %%
