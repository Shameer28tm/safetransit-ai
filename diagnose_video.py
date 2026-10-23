"""
Diagnostic utility for SafeTransit-AI computer-vision pipeline.
Reports per-frame:
- Frame index and timestamp
- Raw output tensor shape
- Candidate class IDs
- Confidence scores before and after filtering
- Final detections and bounding boxes
"""

import os
import sys
import argparse
import cv2
import numpy as np
from typing import Dict, Any

from yolo_detector import detector

def diagnose_video(video_path: str, conf_threshold: float = 0.15, max_frames: int = 150):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"============================================================", flush=True)
    print(f" SafeTransit-AI Diagnostic Inspection: {os.path.basename(video_path)}", flush=True)
    print(f" Dimensions: {w}x{h} | FPS: {fps:.2f} | Total Frames: {total_frames}", flush=True)
    print(f" Active Backend: {detector.backend} | Confidence Threshold: {conf_threshold}", flush=True)
    print(f"============================================================", flush=True)

    frame_idx = 0
    candidate_summary = []

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames > 0 and frame_idx >= max_frames):
            break

        timestamp_sec = frame_idx / fps
        timestamp_str = f"{int(timestamp_sec // 60):02d}:{timestamp_sec % 60:05.2f}"

        # 1. Letterbox & Raw Tensor
        input_img, r, (dw, dh) = detector.letterbox(frame, new_shape=(640, 640))
        blob = cv2.dnn.blobFromImage(input_img, scalefactor=1.0 / 255.0, size=(640, 640), swapRB=True, crop=False)
        detector.net.setInput(blob)
        outputs = detector.net.forward()

        tensor_shape = list(outputs.shape)  # e.g. [1, 6, 8400]
        preds = outputs[0].T  # [8400, 6]

        # Raw scores across grid
        max_car_raw = float(np.max(preds[:, 4]))
        max_acc_raw = float(np.max(preds[:, 5]))

        # Post-processed detections
        detections = detector.detect(frame, conf_threshold=conf_threshold)
        acc_detections = [d for d in detections if d['label'] == 'accident']

        if acc_detections or max_acc_raw >= 0.05 or frame_idx % 20 == 0:
            print(f"\n[Frame {frame_idx:04d} | {timestamp_str}] Raw Tensor Shape: {tensor_shape}", flush=True)
            print(f"  Raw Max Confidence: Car (Class 0) = {max_car_raw:.4f} | Accident (Class 1) = {max_acc_raw:.4f}", flush=True)
            print(f"  Detections >= {conf_threshold}: {len(detections)} (Accidents: {len(acc_detections)})", flush=True)
            for d in detections:
                print(f"    -> [{d['label'].upper()}] Conf: {d['confidence']:.4f} | Box: {d['box']}", flush=True)

        if acc_detections:
            candidate_summary.append({
                "frame": frame_idx,
                "timestamp": timestamp_str,
                "detections": acc_detections
            })

        frame_idx += 1

    cap.release()

    print(f"\n============================================================", flush=True)
    print(f" DIAGNOSTIC SUMMARY", flush=True)
    print(f" Analyzed {frame_idx} frames.", flush=True)
    print(f" Total accident frames flagged at conf >= {conf_threshold}: {len(candidate_summary)}", flush=True)
    if candidate_summary:
        print(" Peak Accident Events:", flush=True)
        for ev in candidate_summary[:5]:
            for d in ev['detections']:
                print(f"   {ev['timestamp']} (Frame {ev['frame']}): {d['confidence']*100:.1f}% confidence, Box: {d['box']}", flush=True)
    else:
        print(" [NOTE] No accident detections passed the current threshold.", flush=True)
    print(f"============================================================", flush=True)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "static/uploaded_video.mp4"
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    diagnose_video(target, conf_threshold=thresh)
