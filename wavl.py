import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray

# Load an example grayscale image
original_image = cv2.imread('test.jpg', 0)

# Perform 2D wavelet transform (using Daubechies wavelet family)
coeffs = pywt.wavedec2(original_image, 'db1', level=3)

# Extract the approximation and detail coefficients
cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

# Visualize the original image and its wavelet coefficients
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
ax = axes.ravel()
ax[0].imshow(original_image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(cA3, cmap='gray')
ax[1].set_title('Approximation Coefficients (cA3)')
ax[2].imshow(cH3, cmap='gray')
ax[2].set_title('Horizontal Detail Coefficients (cH3)')
ax[3].imshow(cV3, cmap='gray')
ax[3].set_title('Vertical Detail Coefficients (cV3')
#ax[4].imshow(cD3, cmap='gray')
#ax[4].set_title('Diagonal Detail Coefficients (cD3)')
plt.show()