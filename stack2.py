import cv2
import numpy as np
import matplotlib.pyplot as plt


def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def adjust_brightness_contrast(image, alpha, beta):
    return cv2.addWeighted(image, alpha, image, 0, beta)

image = cv2.imread('test3_cr.jpg')
enhanced_image = adjust_brightness_contrast(image, 1.2, 30)
enhanced_image = sharpen_image(image)
original_and_enhanced_image = np.hstack((image, enhanced_image))

plt.figure(figsize = [30, 30])
plt.axis('off')
plt.imshow(original_and_enhanced_image [:,:,::-1])
plt.show()