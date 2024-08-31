from ultralytics import YOLO

# 4.1 Load the trained YOLOv8 model
# Replace 'path_to_your_trained_model.pt' with the actual path to your trained model file.
model = YOLO('yolov8n.pt')

# 4.2 Evaluate the trained model on the validation set
# This will calculate metrics such as precision, recall, and mAP on your validation data.
metrics = model.val()

# Print evaluation metrics
print(f"Validation Results:\n Precision: {metrics['precision']}\n Recall: {metrics['recall']}\n mAP50: {metrics['map50']}\n mAP50-95: {metrics['map']}")

# 4.3 Export the trained model to a format suitable for deployment (e.g., ONNX)
# This allows you to deploy the model on edge devices or integrate it with other systems.
export_path = model.export(format='onnx')  # Export model to ONNX format
print(f"Model exported to {export_path}")
