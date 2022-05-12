import matplotlib.pyplot as plt
from matplotlib import image

import numpy as np

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

original = image.imread("img\\dice1.jpg")
gray = rgb2gray(original)

gray = gray * 255
gray = gray.astype(np.int32)

tresh = threshold_otsu(gray)

binary = gray > tresh

# plot
fig, axes = plt.subplots(ncols=3)
axs = axes.ravel()

axs[0].imshow(gray, cmap='gray', vmin=0, vmax=255)
axs[1].imshow(binary, cmap='gray', vmin=0, vmax=1)

plt.show()