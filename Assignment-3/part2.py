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

a = 0.073235
b = 0.176765

kernel = np.array([[a,b,a],[b,0,b],[a,b,a]])
print(np.sum(kernel))
def inpaint(mask, img, kernel, max_iter = 20000):

    R,G,B = np.array_split(cv.cvtColor(img, cv.COLOR_BGR2RGB), 3, axis=2)
    R = R[:, :, 0]
    G = G[:,:,0]
    B = B[:, :, 0]

    d1 = len(R)
    d2 = len(R[0])

    R_avg = int(np.sum(R)/(d1*d2))
    B_avg = int(np.sum(B)/(d1*d2))
    G_avg = int(np.sum(G)/(d1*d2))
    print(R_avg,G_avg,B_avg)
    # temp_R = mask/255*R_avg + (1-mask/255)*R
    # temp_G = mask/255*G_avg + (1-mask/255)*G
    # temp_B = mask/255*B_avg + (1-mask/255)*B

    temp_R = R
    temp_G = G
    temp_B = B

    final2 = np.stack((temp_B,temp_G,temp_R), axis=2).astype("uint8")
    cv.imshow("f", final2)

    for itr in range(max_iter):
        print(itr)
        cv.filter2D(src=temp_R, ddepth=-1, kernel=kernel)
        cv.filter2D(src=temp_G, ddepth=-1, kernel=kernel)
        cv.filter2D(src=temp_B, ddepth=-1, kernel=kernel)
        temp_R = mask/255*cv.filter2D(src=temp_R, ddepth=-1, kernel=kernel) + (1-mask/255)*R
        temp_G = mask/255*cv.filter2D(src=temp_G, ddepth=-1, kernel=kernel) + (1-mask/255)*G
        temp_B = mask/255*cv.filter2D(src=temp_B, ddepth=-1, kernel=kernel) + (1-mask/255)*B

    final = np.stack((temp_B,temp_G,temp_R), axis=2).astype("uint8")
    return final
    

path1 = 'COL783_Test_Dataset\COL783_Test_Dataset\Distorted\line_Image_0000.png'
img = cv.imread(path1)
path2 = 'COL783_Test_Dataset\COL783_Test_Dataset\Masks\line_Image_0000.png'
mask = cv.imread(path2,0)

result = inpaint(mask,img,kernel)
cv.imshow("result", result)
cv.waitKey(0)

