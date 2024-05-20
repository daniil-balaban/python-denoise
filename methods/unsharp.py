import os
import cv2
import matplotlib.pyplot as plt

import time

path = os.path.dirname(os.path.realpath(__file__))
obj = 'vert'
filePath = f'{path}\\..\\{obj}\\{obj}'
img = cv2.imread(f'{filePath}_non-local-denoise.jpg')

start = time.time()
# Sharpen using unsharp + Gaussian blur
sigmaX = 1
sharpened_image = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (0, 0), sigmaX), -1, 0)

b,g,r = cv2.split(sharpened_image)   # get b,g,r
sharpened_image = cv2.merge([r,g,b]) # switch it to rgb

alpha = 2  # Contrast control (1.0 means no change)
beta = -15  # Brightness control (0 means no change, positive values increase brightness)

# Apply brightness and contrast
brightened_image = cv2.convertScaleAbs(sharpened_image, alpha = alpha, beta = beta)
print(time.time() - start)

plt.imsave(f'{filePath}_sharp+bright.jpg', brightened_image)
