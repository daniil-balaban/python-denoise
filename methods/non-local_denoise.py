import os
import cv2
import matplotlib.pyplot as plt

import time

path = os.path.dirname(os.path.realpath(__file__))
obj = 'vert'
filePath = f'{path}\\..\\{obj}\\{obj}'
img = cv2.imread(f'{filePath}.jpg')

start = time.time()

arr = [20,7,21]
denoised_image = cv2.fastNlMeansDenoising(img,None,arr[0],arr[1],arr[2])

b,g,r = cv2.split(denoised_image)   # get b,g,r
denoised_image = cv2.merge([r,g,b]) # switch it to rgb
print(time.time() - start)

plt.imsave(f'{filePath}_non-local-denoise.jpg', denoised_image)
