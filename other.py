from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("yolov8n.pt")  # Load a pretrained model

# Load a local image file
image_path = 'img.jpg'  # Path to the local image
original_img = cv2.imread(image_path)

# Check if the image was loaded correctly
if original_img is None:
    raise Exception("Failed to load image. The image file is empty or cannot be read.")

# Use the model to get detection results
results = model(original_img)  # Predict on the loaded image

# Create a mask for non-human objects
mask = np.ones(original_img.shape, dtype=np.uint8) * 255

# Process each detection result
for detection in results[0].boxes:
    cls = int(detection.cls.item())  # Convert tensor to Python scalar
    if cls == 0:  # Assuming class 0 is 'person' (verify your model's class IDs if necessary)
        # Extract bounding box coordinates
        xyxy = detection.xyxy.cpu().numpy().astype(int)  # Convert tensor to numpy array and then to integers
        x1, y1, x2, y2 = xyxy[0]  # Get coordinates
        # Mask the area where the person is detected
        cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), thickness=cv2.FILLED)

# Apply the mask to the original image
result_img = cv2.bitwise_and(original_img, mask)

# Convert BGR to RGB for display
result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

# Show the result
cv2.imshow('Image with Humans Removed', result_img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(results)
