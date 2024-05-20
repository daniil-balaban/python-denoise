from matplotlib import pyplot as plt
import os

path = os.path.dirname(os.path.realpath(__file__))

images = [
    (plt.imread(f'{path}\\koala.jpg'), "Default img"),
    (plt.imread(f'{path}\\koala_ph.jpg'), "Photoshop noise reduction + brightnes"),
    (plt.imread(f'{path}\\koala_non-local-denoise.jpg'), "fastNlMeansDenoising"),
    (plt.imread(f'{path}\\koala_sharp+bright.jpg'), "unsharp sharp + brightness"),
    (plt.imread(f'{path}\\koala_nvd.jpg'), "koala after nvidia nn"),
]

plt.figure(figsize=(15, 10))
for i,img in enumerate(images):
    plt.subplot(1, len(images)+1, i+1)
    plt.imshow(img[0])
    plt.title(img[1])

plt.show()