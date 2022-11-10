import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import exposure
from scipy.signal import argrelextrema
import seaborn as sns
from scipy import stats
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.signal import correlate2d
from math import sqrt, pi
from scipy.signal import medfilt
from scipy import ndimage

def inpaint(mask, img, kernel, max_iter = 100):
    R,G,B = np.array_split(cv.cvtColor(img, cv.COLOR_BGR2RGB), 3, axis=2)
    R = R[:, :, 0]
    G = G[:,:,0]
    B = B[:, :, 0]

    d1 = len(R)
    d2 = len(R[0])

    R_avg = sum(R)/(d1*d2)
    B_avg = sum(B)/(d1*d2)
    G_avg =sum(G)/(d1*d2)

    temp_R = mask/255*R + (1-mask/255)*R_avg
    temp_G = mask/255*G + (1-mask/255)*G_avg
    temp_B = mask/255*B + (1-mask/255)*B_avg


    final = np.stack((temp_B,temp_G,temp_R), axis=2)
    kernel3ch = cv.cvtColor(kernel, cv.COLOR_GRAY2BGR)
    b_size = len(kernel)//2
    with_border = cv.copyMakeBorder(final,b_size, b_size, b_size, b_size, cv.BORDER_REPLICATE)

    ch = 3
    for itr in range(max_iter):
        with_border = cv.copyMakeBorder(final,b_size, b_size, b_size, b_size, cv.BORDER_REPLICATE)
        for r in range(final.shape[0]):


            for c in range(final.shape[1]):
                



