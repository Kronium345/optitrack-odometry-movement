import cv2
from ultralytics import YOLO

# 5.1 Load the trained model (in .pt format or exported format like .onnx)
model = YOLO('yolov8n.pt')

# 5.2 Load an image or video for inference
# For an image:
image_path = 'dataset/images/val/13_06_30_375467.png'
image = cv2.imread(image_path)

# Run inference
results = model(image)

# 5.3 Display results
# Draw bounding boxes on the image
annotated_image = results.render()[0]  # Get the annotated image

# Show the image using OpenCV
cv2.imshow("YOLOv8 Inference", annotated_image)
cv2.waitKey(0)  # Press any key to close the image window

# Optional: Save the annotated image
cv2.imwrite('annotated_image.jpg', annotated_image)

