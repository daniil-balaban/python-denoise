import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)


path = os.path.dirname(os.path.realpath(__file__))
obj = 'car'
filePath = f'{path}\\..\\{obj}\\{obj}'

# Load a good quality picture
image = cv2.imread(f'{filePath}1.jpg')

# Add Gaussian noise to the image
b,g,r = cv2.split(add_gaussian_noise(image))   # get b,g,r
noisy_image = cv2.merge([r,g,b]) # switch it to rgb

plt.imsave(f'{filePath}_gauss-noise.jpg', noisy_image)
