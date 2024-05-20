import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, color
from skimage import img_as_ubyte
from skimage.util import view_as_blocks

def rle_encode(diff):
    diff_flat = diff.flatten()
    rle_result = []
    count = 1
    for i in range(1, len(diff_flat)):
        if diff_flat[i] == diff_flat[i - 1]:
            count += 1
        else:
            rle_result.extend([diff_flat[i - 1], count])
            count = 1
    rle_result.extend([diff_flat[-1], count])
    return rle_result

# Завантаження та конвертація зображення в відтінки сірого
image = io.imread("lamp.jpg")
gray_image = color.rgb2gray(image)

# Перетворення зображення у вектор для кластеризації
data = gray_image.reshape((-1, 1))

# Кластеризація методом K-Means (2 класи - фон та об'єкт)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)
segmented_image = kmeans.labels_.reshape(gray_image.shape)

# Кодування диференціалів для стискання
differential_coding = np.diff(segmented_image.flatten(), axis=0)
rle_encoded = rle_encode(differential_coding)

# Відобразимо результати
plt.figure(figsize=(12, 4))

plt.subplot(141)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')

plt.subplot(142)
plt.imshow(segmented_image, cmap='viridis')
plt.title('Segmented Image (K-Means)')

plt.subplot(143)
plt.plot(differential_coding, color='blue')
plt.title('Differential Coding')

plt.subplot(144)
plt.plot(rle_encoded, color='blue')
plt.title('RLE Encoded Differential Coding')
plt.show()
