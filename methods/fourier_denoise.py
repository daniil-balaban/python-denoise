import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import time

path = os.path.dirname(os.path.realpath(__file__))
obj = 'vert'
filePath = f'{path}\\..\\{obj}\\{obj}'
img = cv2.imread(f'{filePath}.jpg',  0)

start = time.time()

f_transform = np.fft.fft2(img)
center_shift = np.fft.fftshift(f_transform)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

center_shift[crow - 4:crow + 4, 0:ccol - 10] = 1
center_shift[crow - 4:crow + 4, ccol + 10:] = 1

f_shift = np.fft.ifftshift(center_shift)
denoised_image = np.fft.ifft2(f_shift)
denoised_image = np.abs(denoised_image)
print(time.time() - start)

plt.imsave(f'{filePath}-fourier-denoise.jpg', denoised_image, cmap="grey")
