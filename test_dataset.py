import cv2
import glob
from ultralytics import YOLO

model = YOLO("best.pt")
print("Model names:", model.names)

val_images = glob.glob("testtech/images/validation/*.jpg")[:15]
print(f"Testing on {len(val_images)} validation images:")

for img_p in val_images:
    res = model.predict(img_p, conf=0.1, verbose=False)
    boxes = res[0].boxes
    if len(boxes) > 0:
        print(f"  {img_p}: {len(boxes)} detection(s)")
        for b in boxes:
            c = int(b.cls[0].cpu().numpy())
            score = float(b.conf[0].cpu().numpy())
            print(f"    -> Class: {model.names[c]} ({c}), Conf: {score:.3f}")
    else:
        print(f"  {img_p}: 0 detections at conf=0.1")
