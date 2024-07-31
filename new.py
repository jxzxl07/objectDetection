from ultralytics import YOLO
import cv2
import numpy as np

# Load the model
model = YOLO("yolov8n.pt")  # Load a pretrained model

# Load a local image file
original_img = cv2.imread('img.jpg')

# Use the model to get detection results
results = model(original_img)  # Predict on the loaded image

# Create a mask for the detected humans
mask = np.zeros(original_img.shape[:2], dtype=np.uint8)

# Process each detection result
for detection in results[0].boxes:
    cls = int(detection.cls.item())  # Convert tensor to Python scalar
    if cls == 0:  # Assuming class 0 is 'person' (verify your model's class IDs if necessary)
        # Extract bounding box coordinates
        xyxy = detection.xyxy.cpu().numpy().astype(int)  # Convert tensor to numpy array and then to integers
        x1, y1, x2, y2 = xyxy[0]  # Get coordinates
        
        # Fill the mask for detected humans
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255), thickness=cv2.FILLED)

# Create an inpainting mask where humans are present
inpainting_mask = cv2.bitwise_not(mask)

# Use inpainting to fill the human areas
result_img = cv2.inpaint(original_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Convert BGR to RGB for display
result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

# Show the result
cv2.imshow('Image with Humans Removed', result_img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(results)
