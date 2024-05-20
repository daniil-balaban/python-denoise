from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Завантажимо приклад даних (зображень рукописних цифр)
digits = load_digits()
data = digits.data

# Застосуємо алгоритм кластеризації KMeans
kmeans = KMeans(n_clusters=10, random_state=42, algorithm='elkan')
clusters = kmeans.fit_predict(data)

# Відобразимо кілька зображень з кожного кластера
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
for i in range(10):
    ax[i//5, i%5].imshow(data[clusters == i][0].reshape(8, 8), cmap='gray')
    ax[i//5, i%5].axis('off')
plt.show()
