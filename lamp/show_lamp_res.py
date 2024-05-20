from matplotlib import pyplot as plt
import os

path = os.path.dirname(os.path.realpath(__file__))

images = [
    (plt.imread(f'{path}\\lamp.jpg'), "Default img"),
    (plt.imread(f'{path}\\lamp_non-local-denoise.jpg'), "fastNlMeansDenoising"),
    (plt.imread(f'{path}\\lamp_sharp+bright.jpg'), "unsharp + brightness")
]

plt.figure(figsize=(15, 10))
for i,img in enumerate(images):
    plt.subplot(1, len(images)+1, i+1)
    plt.imshow(img[0])
    plt.title(img[1])

plt.show()