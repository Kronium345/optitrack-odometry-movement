from ultralytics import YOLO

# 3.1 Load a pre-trained YOLOv8 model
# Here, 'yolov8n.pt' is the small, fast model variant. You can change this to 'yolov8s.pt', etc., for larger models.
model = YOLO('yolov8n.pt')

# 3.2 Train the model on your dataset
# You will need to specify the path to your 'data.yaml' file, which contains information about your dataset.
# Adjust the 'epochs', 'batch', and 'imgsz' parameters as needed.

model.train(
    data='data.yaml',  
    epochs=50,                      # Number of training epochs
    imgsz=640,                       # Image size (pixels)
    batch=8,                        # Batch size
    device='cpu'                       # Specify GPU ('0' for the first GPU, or 'cpu' for CPU)
)

print("Training complete.")

