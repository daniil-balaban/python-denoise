import cv2
import matplotlib.pyplot as plt

# Завантаження зображення (приклад)
image = cv2.imread('lamp.jpg', cv2.IMREAD_GRAYSCALE)

# Застосування фільтра Гаусса для зменшення шуму
filtered_image = cv2.GaussianBlur(image, (5, 5), 0)

# Відображення результатів
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Оригінальне зображення')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Зменшення шуму за допомогою фільтра Гаусса')

plt.show()
