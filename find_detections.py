import cv2
import glob
import numpy as np

net = cv2.dnn.readNetFromONNX("best.onnx")

def test_frame(img, conf_thresh=0.01):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (640, 640)), 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()[0].T # [8400, 6]
    scores = out[:, 4:]
    max_c = np.max(scores[:, 0])
    max_a = np.max(scores[:, 1])
    return max_c, max_a

print("1. Scanning demo_media/highway_collision_demo.webm...")
cap = cv2.VideoCapture("demo_media/highway_collision_demo.webm")
frame_idx = 0
max_car = 0.0
max_acc = 0.0
best_frame_car = None
best_frame_acc = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 15 == 0:
        c, a = test_frame(frame)
        if c > max_car:
            max_car = c
            best_frame_car = frame_idx
        if a > max_acc:
            max_acc = a
            best_frame_acc = frame_idx
    frame_idx += 1
cap.release()

print(f"Highway Demo: Peak car score = {max_car:.4f} (frame {best_frame_car}), Peak accident score = {max_acc:.4f} (frame {best_frame_acc})")

print("\n2. Scanning cr.mp4...")
cap = cv2.VideoCapture("cr.mp4")
frame_idx = 0
max_car = 0.0
max_acc = 0.0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 10 == 0:
        c, a = test_frame(frame)
        if c > max_car:
            max_car = c
        if a > max_acc:
            max_acc = a
    frame_idx += 1
cap.release()
print(f"cr.mp4: Peak car score = {max_car:.4f}, Peak accident score = {max_acc:.4f}")

print("\n3. Scanning testtech training set...")
train_imgs = glob.glob("testtech/images/training/*.jpg")[:30]
train_max_car = 0.0
train_max_acc = 0.0
for p in train_imgs:
    img = cv2.imread(p)
    if img is not None:
        c, a = test_frame(img)
        if c > train_max_car:
            train_max_car = c
        if a > train_max_acc:
            train_max_acc = a
print(f"Training set sample (30 images): Peak car = {train_max_car:.4f}, Peak accident = {train_max_acc:.4f}")
