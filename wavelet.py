import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import sobel
import matplotlib.pyplot as plt
import pywt
import cv2

# Load a sample image
original_image = cv2.imread('test.jpg')
image = rgb2gray(original_image)

# Perform 2D wavelet transform
coeffs2 = pywt.dwt2(image, 'haar')
cA, (cH, cV, cD) = coeffs2

# Edge detection using Sobel operator on the approximation coefficient (cA)
edge_sobel = sobel(cA)

# Feature extraction using the detail coefficients (cH, cV, cD)
# You can apply your custom feature extraction method here, for example, by analyzing the statistics of the coefficients

# Visualize the results
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')

ax[1].imshow(edge_sobel, cmap='gray')
ax[1].set_title('Edge Detection (Sobel)')

# Visualize the wavelet coefficients
ax[2].imshow(np.hstack((cA, cH, cV, cD)), cmap='gray')
ax[2].set_title('Wavelet Coefficients')

plt.show()
