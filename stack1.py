import cv2
import matplotlib.pyplot as plt

image = cv2.imread('frame0.jpg')
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
out_gray=cv2.divide(image, bg, scale=255)
out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 

plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.imshow(out_gray, cmap='gray')
plt.title('Gray')

plt.subplot(1, 3, 2)
plt.imshow(out_binary, cmap='gray')
plt.title('Binary')

plt.subplot(1, 3, 3)
plt.imshow(image, cmap='gray')
plt.title('Default')

plt.show()


#cv2.imshow('binary', out_binary)  
cv2.imwrite('binary.png',out_binary)

#cv2.imshow('gray', out_gray)  
cv2.imwrite('gray.png',out_gray)
input()