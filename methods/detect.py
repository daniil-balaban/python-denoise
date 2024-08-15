import tensorflow as tf
import tensorflow_hub as hub

# Load the SSD MobileNet model from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")


import numpy as np
import cv2

# Load the image
img_path = 'C:\\Users\\rss\\uni\\python-denoise\\lamp\\lamp-sharp+bright.jpg'
#image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(img_path)
#image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image_resized = cv2.resize(image, (320, 320))  # Model expects 320x320 images
image_np = np.expand_dims(image_resized, axis=0)


# Make predictions
results = model(image_np)

# Extract detection boxes, class IDs, and scores
detection_boxes = results['detection_boxes'][0].numpy()
detection_classes = results['detection_classes'][0].numpy().astype(np.int64)
detection_scores = results['detection_scores'][0].numpy()

# Define the COCO class ID for cars
CAR_CLASS_ID = 3

# Set a threshold for detection confidence
confidence_threshold = 0.6

# Loop through the detections and draw bounding boxes for cars
for i in range(len(detection_boxes)):
    if detection_scores[i] > confidence_threshold:
        box = detection_boxes[i]
        start_point = (int(box[1] * image.shape[1]), int(box[0] * image.shape[0]))
        end_point = (int(box[3] * image.shape[1]), int(box[2] * image.shape[0]))
        cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)  # Draw bounding box
        print(detection_classes[i])

# Display the image with detections
cv2.imshow("Lamp Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
