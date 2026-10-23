from ultralytics import YOLO

# Load your trained model
model = YOLO("C:/Users/Admin/OneDrive/Desktop/MajorProjecr/ACCIDENT DETECTION/best.pt")  # your trained YOLOv8 model

# Evaluate on the validation set
metrics = model.val(data="C:/Users/Admin/OneDrive/Desktop/MajorProjecr/ACCIDENT DETECTION/data.yaml")

# Print mAP
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
