import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class YOLODetector:
    """
    Robust Multi-Backend YOLOv8 Object & Accident Detector for SafeTransit-AI.
    
    Backends:
    - 'opencv_dnn': Native C++ OpenCV DNN inference over best.onnx (immune to Windows PyTorch DLL issues).
    - 'torch': Ultralytics YOLOv8 inference over best.pt (used in Linux/WSL2 when PyTorch is available).
    
    Provides uniform API: detect(frame) -> List[Dict] with standardized keys:
      'box': [x1, y1, x2, y2]
      'confidence': float
      'class_id': int
      'label': str
    """

    def __init__(
        self,
        onnx_model_path: str = "best.onnx",
        pt_model_path: str = "best.pt",
        classes_path: str = "coco1.txt",
        preferred_backend: Optional[str] = None
    ):
        self.onnx_model_path = onnx_model_path
        self.pt_model_path = pt_model_path
        self.classes_path = classes_path
        self.preferred_backend = preferred_backend or os.environ.get("INFERENCE_BACKEND", "opencv_dnn").lower()

        self.class_names = self._load_classes()
        self.backend = None
        self.net = None
        self.torch_model = None
        self.is_ready = False
        self.status_message = ""

        self._initialize_backend()

    def _load_classes(self) -> List[str]:
        if os.path.exists(self.classes_path):
            try:
                with open(self.classes_path, "r", encoding="utf-8") as f:
                    classes = [line.strip() for line in f.read().splitlines() if line.strip()]
                if classes:
                    return classes
            except Exception as e:
                print(f"[YOLODetector] Error reading classes from {self.classes_path}: {e}")
        # Default fallback
        return ["car", "accident"]

    def _initialize_backend(self):
        # Priority 1: Check OpenCV DNN if preferred or if ONNX exists
        if self.preferred_backend in ("opencv_dnn", "onnx") or os.path.exists(self.onnx_model_path):
            if os.path.exists(self.onnx_model_path):
                try:
                    print(f"[YOLODetector] Loading ONNX model into OpenCV DNN: {self.onnx_model_path}...")
                    self.net = cv2.dnn.readNetFromONNX(self.onnx_model_path)
                    # Use OpenCV DNN default backend
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.backend = "opencv_dnn"
                    self.is_ready = True
                    self.status_message = f"OpenCV DNN active with {self.onnx_model_path}"
                    print(f"[YOLODetector] {self.status_message}")
                    return
                except Exception as e:
                    print(f"[YOLODetector] Failed to initialize OpenCV DNN backend: {e}")

        # Priority 2: Check PyTorch / Ultralytics
        if self.preferred_backend in ("torch", "ultralytics") or not self.is_ready:
            try:
                from ultralytics import YOLO
                print(f"[YOLODetector] Attempting to load PyTorch model: {self.pt_model_path}...")
                self.torch_model = YOLO(self.pt_model_path)
                self.backend = "torch"
                self.is_ready = True
                self.status_message = f"Ultralytics PyTorch active with {self.pt_model_path}"
                print(f"[YOLODetector] {self.status_message}")
                return
            except Exception as e:
                print(f"[YOLODetector] PyTorch / Ultralytics runtime unavailable: {e}")

        # Priority 3: Fallback warning
        if not self.is_ready:
            self.backend = "unavailable"
            self.status_message = (
                f"Detection models not yet available. "
                f"Expected '{self.onnx_model_path}' for OpenCV DNN or working PyTorch runtime for '{self.pt_model_path}'."
            )
            print(f"[YOLODetector] Warning: {self.status_message}")

    @staticmethod
    def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Resize and pad image while meeting stride-multiple constraints."""
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2.0  # divide padding into 2 sides
        dh /= 2.0

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Run inference on frame and return list of detections.
        """
        if not self.is_ready or frame is None:
            return []

        if self.backend == "opencv_dnn":
            return self._detect_opencv_dnn(frame, conf_threshold, nms_threshold)
        elif self.backend == "torch":
            return self._detect_torch(frame, conf_threshold)
        return []

    def _detect_opencv_dnn(
        self,
        frame: np.ndarray,
        conf_threshold: float,
        nms_threshold: float
    ) -> List[Dict[str, Any]]:
        orig_h, orig_w = frame.shape[:2]
        
        # 1. Preprocess with letterbox to 640x640
        input_img, r, (dw, dh) = self.letterbox(frame, new_shape=(640, 640))
        blob = cv2.dnn.blobFromImage(input_img, scalefactor=1.0 / 255.0, size=(640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # 2. Forward pass
        outputs = self.net.forward()
        
        # Output shape for YOLOv8: [1, 4 + num_classes, 8400]
        if len(outputs.shape) == 3:
            preds = outputs[0]  # [4 + num_classes, 8400]
            preds = preds.T     # [8400, 4 + num_classes]
        else:
            return []

        boxes = []
        confidences = []
        class_ids = []

        num_classes = len(self.class_names)
        
        for row in preds:
            class_scores = row[4:4 + num_classes]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence >= conf_threshold:
                cx, cy, w, h = row[0:4]
                # Map coordinates back from letterboxed 640x640 to original frame
                x1 = (cx - w / 2.0 - dw) / r
                y1 = (cy - h / 2.0 - dh) / r
                x2 = (cx + w / 2.0 - dw) / r
                y2 = (cy + h / 2.0 - dh) / r

                # Clip to frame boundary
                x1 = max(0, min(int(round(x1)), orig_w - 1))
                y1 = max(0, min(int(round(y1)), orig_h - 1))
                x2 = max(0, min(int(round(x2)), orig_w - 1))
                y2 = max(0, min(int(round(y2)), orig_h - 1))
                
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)

                boxes.append([x1, y1, box_w, box_h])
                confidences.append(confidence)
                class_ids.append(class_id)

        if not boxes:
            return []

        # 3. Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=conf_threshold, nms_threshold=nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = boxes[idx]
                cid = class_ids[idx]
                label = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
                detections.append({
                    "box": [bx, by, bx + bw, by + bh],
                    "confidence": round(confidences[idx], 4),
                    "class_id": cid,
                    "label": label
                })

        return detections

    def _detect_torch(self, frame: np.ndarray, conf_threshold: float) -> List[Dict[str, Any]]:
        detections = []
        try:
            results = self.torch_model.predict(frame, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy())
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                detections.append({
                    "box": xyxy,
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "label": label
                })
        except Exception as e:
            print(f"[YOLODetector] Torch inference error: {e}")
        return detections

# Global detector instance
detector = YOLODetector()
