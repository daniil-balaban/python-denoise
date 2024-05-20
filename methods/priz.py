from skimage import io, color, feature
import matplotlib.pyplot as plt

# Завантаження та конвертація зображення в відтінки сірого
image = io.imread("test.jpg")
gray_image = color.rgb2gray(image)

# Виведення гістограми зображення
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.hist(gray_image.ravel(), bins=256, color='gray', histtype='step')
plt.title('Image Histogram')

# Виведення контурів зображення
edges = feature.canny(gray_image)
plt.subplot(132)
plt.imshow(edges, cmap='gray')
plt.title('Image Contours')

# Виведення картинки текстурних характеристик (Local Binary Pattern)
lbp = feature.local_binary_pattern(gray_image, P=8, R=1)
plt.subplot(133)
plt.imshow(lbp, cmap='gray')
plt.title('Local Binary Pattern')

plt.show()
