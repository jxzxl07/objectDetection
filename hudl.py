from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model("https://ultralytics.com/images/Uc2XsXJ.jpg")  # predict on an image

# Visualize the detection
img = results[0].plot()  # This plots the detections on the image

# Convert BGR to RGB (OpenCV uses BGR by default)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('Detection Results', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(results)