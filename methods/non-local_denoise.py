import os
import cv2
import matplotlib.pyplot as plt

import time

path = os.path.dirname(os.path.realpath(__file__))
obj = 'koala'
filePath = f'{path}\\..\\{obj}\\{obj}'
img = cv2.imread(f'{filePath}.jpg')

#start = time.time()

#l_a = [20,7,21]
#k_a = [21,7,20]

r = range(1, 51)

for x in r:
    for y in r:
        for z in r:
            denoised_image = cv2.fastNlMeansDenoising(img, None, x, y, z)
            denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
            plt.imsave(f'{filePath}\\koala-non-local-denoise-test-{x}-{y}-{z}.jpg', denoised_image)


#denoised_image = cv2.fastNlMeansDenoising(img, None, k_a[0], k_a[1], k_a[2])

#b,g,r = cv2.split(denoised_image)   # get b,g,r
#denoised_image = cv2.merge([r,g,b]) # switch it to rgb
#print(time.time() - start)

#plt.imsave(f'{filePath}-non-local-denoise-test-{k_a[0]}-{k_a[1]}-{k_a[2]}.jpg', denoised_image)
