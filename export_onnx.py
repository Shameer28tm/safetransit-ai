import sys
from ultralytics import YOLO

print("Loading model best.pt...")
model = YOLO("best.pt")
print("Exporting model to ONNX...")
exported_path = model.export(format="onnx", imgsz=640, opset=12)
print("SUCCESS: Model exported to:", exported_path)
