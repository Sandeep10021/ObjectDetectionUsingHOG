import sys
sys.path.append("E:/M-Tech/Projects 2021/AMSIP")
from hog_amsip import *
import matplotlib.pyplot as plt
from skimage import io
from skimage import data, color, exposure
from PIL import Image

img = io.imread(r"E:/M-Tech/Projects 2021/AMSIP/shinchan.jpeg")

image = color.rgb2gray(img)

fd, hog_image = hog_nd(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(2,2), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box')
plt.show()


