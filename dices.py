import matplotlib.pyplot as plt
from matplotlib import image

import numpy as np

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening

from os import listdir
from os.path import isfile,  join

def processImg(img):
    # convert to grayscale
    gray = rgb2gray(img)

    # add blur to remove noise
    gray = gaussian(gray, 4)

    # convert to int32
    gray = gray * 255
    gray = gray.astype(np.int32)

    # treshold an binarize
    tresh = threshold_otsu(gray)
    binary = gray > tresh
    binary = opening(binary, disk(4))
    binary = clear_border(binary)

    labeled, amount = label(binary, return_num=True)

    # plot
    fig, axs = plt.subplots(ncols=6, nrows=2, figsize=(10,8), sharex='row', sharey='row')

    #crop image
    for i in range (amount):
        diceXY = np.nonzero(labeled==(i+1))
        sr = min(diceXY[0])
        br = max(diceXY[0])
        sc = min(diceXY[1])
        bc = max(diceXY[1])

        dice = labeled[sr:br, sc:bc]
        dice = 1 - dice

        # show single dice
        axs[1,i].imshow(dice, cmap='gray', vmin=0, vmax=1)
        axs[1,i].set_title("dice %i" % (i+1))
        axs[1,i].axis('off')

    axs[0,0].imshow(img)
    axs[0,0].set_title("original")
    axs[0,0].axis('off')

    axs[0,1].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axs[0,1].set_title("grayscale + gauss")
    axs[0,1].axis('off')

    axs[0,2].imshow(binary, cmap='gray', vmin=0, vmax=1)
    axs[0,2].set_title("tresholded")
    axs[0,2].axis('off')

    axs[0,3].imshow(labeled, cmap='plasma', vmin=0, vmax=amount)
    axs[0,3].set_title("labeled")
    axs[0,3].annotate("N dice: %i" % amount, xy=[0,500], xytext=[0,500], color='w')
    axs[0,3].axis('off')

#imgPath = "img\\"
#images = [image.imread(join(imgPath,f)) for f in listdir(imgPath) if isfile(join(imgPath, f))]
#for i in images:
#    processImg(i)

processImg(image.imread("img\\dice1.jpg"))

plt.show()