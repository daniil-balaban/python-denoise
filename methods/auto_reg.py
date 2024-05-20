import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# Згенеруємо стаціонарний часовий ряд як приклад зображення
np.random.seed(42)
ar_params = [1, -0.7, 0.2]  # Коефіцієнти авторегресії
y = np.zeros(1000)

# Генерація стаціонарного часового ряду
for t in range(2, len(y)):
    y[t] = ar_params[0] + ar_params[1] * y[t-1] + ar_params[2] * y[t-2] + np.random.normal(0, 1)

# Побудова авторегресійної моделі
model = AutoReg(y, lags=[1, 2])  # Вказуємо лаги (затримки)
result = model.fit()

# Побудова графіків
plt.plot(y, label='Original Time Series')
plt.plot(result.predict(start=2, end=len(y)), label='AR Model Prediction')
plt.legend()
plt.show()
