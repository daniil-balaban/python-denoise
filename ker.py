from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np

# Завантажимо попередньо навчену модель VGG16
model = VGG16(weights='imagenet')

# Завантажимо та обробимо зображення
img_path = 'test3_cr.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Отримаємо прогнози класів
predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=3)[0]

# Виведемо результати
print("Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")
