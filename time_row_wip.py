import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from PIL import Image
import cv2

# Створення авторегресійного зображення
np.random.seed(42)
true_image = np.random.random((64, 64))  # Справжнє зображення
image = cv2.imread('lamp_org.jpg', 0)
# Конвертувати в NumPy масив
image_array = np.array(cv2.resize(image, (64, 64)))

# Створення часового ряду з суміші випадкових шумів та стаціонарної складової
time_series = np.cumsum(np.random.normal(0, 0.1, 64 * 64)) + 0.2 * np.sin(np.arange(64 * 64) / 10)

# Додавання стаціонарної складової до часового ряду
time_series += image_array.flatten()

# Моделювання ARIMA для прогнозування часового ряду
model = ARIMA(time_series, order=(1, 0, 0))  # AR(1) модель
result = model.fit()
forecast = result.predict(start=len(time_series), end=len(time_series) + 4096, dynamic=False)

# Повернення до розмірності зображення
generated_image = forecast[-4096:].reshape((64, 64))

# Відображення результатів
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Справжнє зображення')

plt.subplot(1, 2, 2)
plt.imshow(generated_image, cmap='gray')
plt.title('Згенероване зображення за допомогою ARIMA')

plt.show()
