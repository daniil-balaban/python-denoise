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
image = cv2.imread(f'{filePath}.jpg')
noised = cv2.cvtColor(add_gaussian_noise(image), cv2.COLOR_BGR2RGB)

plt.imsave(f'{filePath}-gauss-noise1.jpg', noised)
