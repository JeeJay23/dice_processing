from random import gauss
import matplotlib.pyplot as plt
from matplotlib import image

import numpy as np

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian

original = image.imread("img\\dice2.jpg")
print(original.shape)

# convert to grayscale
gray = rgb2gray(original)

# add blur to remove noise
gray = gaussian(gray, 4)

# convert to int32
gray = gray * 255
gray = gray.astype(np.int32)

# treshold an binarize
tresh = threshold_otsu(gray)
binary = gray > tresh

# plot
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(10,8), sharex='row', sharey='row')

axs[0,0].imshow(original)

axs[0,1].imshow(gray, cmap='gray', vmin=0, vmax=255)

axs[1,0].hist(gray.ravel(), bins=256)
axs[1,0].axvline(tresh, color='r')

axs[0,2].imshow(binary, cmap='gray', vmin=0, vmax=1)

plt.show()