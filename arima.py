import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from scipy import ndimage

# Generate a noisy image
np.random.seed(0)
image = np.random.rand(100, 100)
noisy_image = image + 0.1 * np.random.randn(100, 100)

# Define the ARIMA model
model = ARIMA(noisy_image, order=(5,1,0))  # Example order (p,d,q)

# Fit the model
model_fit = model.fit()

# Generate denoised image using the fitted ARIMA model
denoised_image = model_fit.predict()

# Display the original, noisy, and denoised images
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

plt.subplot(133)
plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image')

plt.show()
