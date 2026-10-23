import cv2
import numpy as np
from ultralytics import YOLO

print("Loading original PyTorch model best.pt...")
model = YOLO("best.pt")
print("Classes:", model.names)

for p in ["scratch/cr.mp4_frame100.jpg", "scratch/test1.mp4_frame100.jpg", "scratch/test2.mp4_frame100.jpg"]:
    img = cv2.imread(p)
    if img is None:
        continue
    results = model.predict(img, conf=0.01, verbose=False)
    boxes = results[0].boxes
    print(f"\n{p} with PyTorch best.pt (conf=0.01):")
    print(f"  Total boxes: {len(boxes)}")
    for b in boxes:
        cls_id = int(b.cls[0].cpu().numpy())
        conf = float(b.conf[0].cpu().numpy())
        xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
        print(f"    Class {cls_id} ({model.names[cls_id]}): conf={conf:.4f}, box={xyxy}")
