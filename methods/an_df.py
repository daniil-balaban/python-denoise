import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

path = os.path.dirname(os.path.realpath(__file__))
obj = 'car'
filePath = f'{path}\\..\\{obj}\\{obj}'

it = 5
delta = 0.5
kappa = 15
ksize = 1


def anisotropic_diffusion(image, iterations=it, delta_t=delta, kappa=kappa):
    img = np.asarray(image, np.float32)

    for _ in range(iterations):
        img_dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        img_dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

        grad_mag = np.sqrt(img_dx**2 + img_dy**2)
        c = 1 / (1 + (grad_mag / kappa)**2)

        diff_x = cv2.Sobel(c * img_dx, cv2.CV_64F, 1, 0, ksize=ksize)
        diff_y = cv2.Sobel(c * img_dy, cv2.CV_64F, 0, 1, ksize=ksize)

        img += delta_t * (diff_x + diff_y)

    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


# Load an image
image = cv2.imread(f'{filePath}-gauss-noise.jpg', cv2.IMREAD_GRAYSCALE)

# Apply anisotropic diffusion
start = time.time()

denoised_image = anisotropic_diffusion(image)

print(time.time() - start)

plt.imsave(f'{filePath}-anisotropic-diffusion-{it}-{delta}-{kappa}-{ksize}.jpg', denoised_image, cmap="grey")
