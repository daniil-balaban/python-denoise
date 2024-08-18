import cv2
import matplotlib.pyplot as plt
import os

path = os.path.dirname(os.path.realpath(__file__))
obj = 'new'
filePath = f'{path}\\..\\{obj}\\{obj}'
image = cv2.imread(f'{filePath}.jpg', -1)
#image = cv2.imread('lamp.jpg', cv2.IMREAD_GRAYSCALE)

# Застосування фільтра Гаусса для зменшення шуму
filtered_image = cv2.GaussianBlur(image, (0, 0), 2)

# Відображення результатів
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Оригінальне зображення')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Зменшення шуму за допомогою фільтра Гаусса')

plt.show()
