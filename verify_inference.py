import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("best.onnx")

for test_img_path in ["scratch/cr.mp4_frame100.jpg", "scratch/test1.mp4_frame100.jpg", "scratch/test2.mp4_frame100.jpg"]:
    img = cv2.imread(test_img_path)
    if img is None:
        continue
    
    h, w = img.shape[:2]
    # Letterbox / resize to 640x640
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (640, 640)), 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    print(f"\n{test_img_path}:")
    print("  Output shape:", out.shape)
    
    # YOLOv8 output: [1, 6, 8400]
    preds = out[0].T # [8400, 6]
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    print("  Max car score:", np.max(scores[:, 0]))
    print("  Max accident score:", np.max(scores[:, 1]))
    print("  Count with score > 0.25:", np.sum(scores > 0.25))
    print("  Count with score > 0.15:", np.sum(scores > 0.15))
