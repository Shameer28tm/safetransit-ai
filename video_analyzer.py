import os
import cv2
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from yolo_detector import detector
from severity_classifier import severity_classifier
from telegram_notifier import notifier

SEVERITY_ORDER = {"fatal": 3, "major": 2, "minor": 1, "low": 1, "medium": 2, "high": 3, "unknown": 0}

class VideoAnalyzer:
    """
    Complete, end-to-end video accident analysis engine for SafeTransit-AI.
    
    Features:
    - File validation (format, size, codec openability)
    - Full-duration multi-frame analysis
    - Temporal incident coalescing (avoids duplicate alerts/records for continuing events)
    - Scene severity estimation with verified 4-feature schema
    - Automatic evidence snapshots saved to static/evidence/
    - Browser-compatible annotated video export (no GUI dependencies)
    - Structured JSON report export
    """

    ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    def __init__(
        self,
        evidence_dir: str = "static/evidence",
        output_video_path: str = "static/processed_video.mp4",
        report_json_path: str = "static/incident_report.json"
    ):
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.output_video_path = output_video_path
        self.report_json_path = report_json_path

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate existence, extension, and non-empty size."""
        if not os.path.exists(file_path):
            return False, f"File does not exist: {file_path}"
        
        ext = Path(file_path).suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"Unsupported video format '{ext}'. Supported formats: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}"
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Uploaded file is empty (0 bytes)."
        
        return True, "Valid video file."

    def analyze_video(
        self,
        video_path: str,
        conf_threshold: float = 0.35,
        target_fps: int = 15,
        trigger_telegram: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze the full video, produce annotated video, evidence snapshots, and incident report.
        """
        valid, err_msg = self.validate_file(video_path)
        if not valid:
            return {
                "success": False,
                "error": err_msg,
                "incidents": [],
                "summary": {}
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"Failed to open video file '{video_path}'. Video codec may be corrupted or unsupported.",
                "incidents": [],
                "summary": {}
            }

        # Video properties
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0 or np.isnan(orig_fps):
            orig_fps = 25.0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1
        
        video_duration_sec = total_frames / orig_fps
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Target dimensions for output video
        out_w = min(1280, max(640, frame_w))
        out_h = int(out_w * (frame_h / frame_w)) if frame_w > 0 else 500
        # Ensure dimensions are even numbers for encoder
        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1

        # Frame step for analysis performance
        step = max(1, int(round(orig_fps / target_fps)))

        # Setup VideoWriter - try mp4v or avc1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = orig_fps / step
        out = cv2.VideoWriter(self.output_video_path, fourcc, out_fps, (out_w, out_h))

        raw_detections_timeline = []
        current_incident = None
        incidents = []
        frame_idx = 0
        analyzed_count = 0
        telegram_alert_result = "No critical accident detected"

        print(f"[VideoAnalyzer] Starting analysis of {video_path}: {total_frames} frames, {video_duration_sec:.1f}s duration.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            analyzed_count += 1
            timestamp_sec = frame_idx / orig_fps
            timestamp_str = f"{int(timestamp_sec // 60):02d}:{timestamp_sec % 60:04.1f}"

            # Run detection
            detections = detector.detect(frame, conf_threshold=conf_threshold)
            accident_dets = [d for d in detections if d['label'].lower() == 'accident']

            # Frame annotation
            annotated_frame = cv2.resize(frame, (out_w, out_h))
            scale_x = out_w / frame_w if frame_w > 0 else 1.0
            scale_y = out_h / frame_h if frame_h > 0 else 1.0

            frame_has_accident = len(accident_dets) > 0
            frame_severity_label = "none"
            frame_severity_info = None

            if frame_has_accident:
                # Calculate features for severity model: [num_detections, avg_confidence, avg_bbox_area, class_id]
                num_detections = len(detections)
                avg_confidence = float(np.mean([d['confidence'] for d in accident_dets]))
                areas = [((d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1])) for d in accident_dets]
                avg_bbox_area = float(np.mean(areas)) if areas else 1000.0
                class_id = accident_dets[0]['class_id']

                frame_severity_info = severity_classifier.predict(
                    num_detections=num_detections,
                    avg_confidence=avg_confidence,
                    avg_bbox_area=avg_bbox_area,
                    class_id=class_id
                )
                frame_severity_label = frame_severity_info['label']

            # Draw all detections
            for d in detections:
                x1, y1, x2, y2 = d['box']
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)

                is_acc = d['label'].lower() == 'accident'
                color = (0, 0, 255) if is_acc else (0, 255, 0)
                thickness = 3 if is_acc else 2

                cv2.rectangle(annotated_frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, thickness)
                
                badge_text = f"{d['label']} {d['confidence']*100:.0f}%"
                if is_acc and frame_severity_label != "none":
                    badge_text += f" [{frame_severity_label.upper()}]"

                # Label background badge
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1_scaled, max(0, y1_scaled - 20)), (x1_scaled + tw + 6, max(0, y1_scaled)), color, -1)
                cv2.putText(annotated_frame, badge_text, (x1_scaled + 3, max(14, y1_scaled - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw Timestamp & Status HUD
            hud_text = f"Time: {timestamp_str} | Active Backend: {detector.backend}"
            cv2.putText(annotated_frame, hud_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Temporal incident tracking & coalescing
            if frame_has_accident:
                peak_conf = max(d['confidence'] for d in accident_dets)

                if current_incident is None:
                    # Start a new incident
                    inc_id = f"inc_{len(incidents) + 1}_{int(timestamp_sec)}"
                    current_incident = {
                        "id": inc_id,
                        "start_sec": timestamp_sec,
                        "end_sec": timestamp_sec,
                        "start_str": timestamp_str,
                        "end_str": timestamp_str,
                        "peak_confidence": round(peak_conf, 4),
                        "peak_severity": frame_severity_label,
                        "severity_info": frame_severity_info,
                        "frame_count": 1,
                        "best_frame": annotated_frame.copy(),
                        "detections_count": len(accident_dets)
                    }
                else:
                    # Check gap between detections: if <= 2.5s, group into same incident
                    gap = timestamp_sec - current_incident["end_sec"]
                    if gap <= 2.5:
                        current_incident["end_sec"] = timestamp_sec
                        current_incident["end_str"] = timestamp_str
                        current_incident["frame_count"] += 1
                        
                        # Update peak confidence & severity
                        if peak_conf > current_incident["peak_confidence"]:
                            current_incident["peak_confidence"] = round(peak_conf, 4)
                            current_incident["best_frame"] = annotated_frame.copy()
                        
                        curr_rank = SEVERITY_ORDER.get(current_incident["peak_severity"], 0)
                        new_rank = SEVERITY_ORDER.get(frame_severity_label, 0)
                        if new_rank > curr_rank:
                            current_incident["peak_severity"] = frame_severity_label
                            current_incident["severity_info"] = frame_severity_info
                    else:
                        # Finalize previous incident
                        self._finalize_incident(current_incident, incidents)
                        # Start new incident
                        inc_id = f"inc_{len(incidents) + 1}_{int(timestamp_sec)}"
                        current_incident = {
                            "id": inc_id,
                            "start_sec": timestamp_sec,
                            "end_sec": timestamp_sec,
                            "start_str": timestamp_str,
                            "end_str": timestamp_str,
                            "peak_confidence": round(peak_conf, 4),
                            "peak_severity": frame_severity_label,
                            "severity_info": frame_severity_info,
                            "frame_count": 1,
                            "best_frame": annotated_frame.copy(),
                            "detections_count": len(accident_dets)
                        }

            out.write(annotated_frame)
            frame_idx += 1

        cap.release()
        out.release()

        # Finalize any pending incident
        if current_incident is not None:
            self._finalize_incident(current_incident, incidents)

        # Trigger single Telegram alert for the most significant incident if requested
        if incidents and trigger_telegram:
            highest_inc = max(incidents, key=lambda x: (SEVERITY_ORDER.get(x["peak_severity"], 0), x["peak_confidence"]))
            success, msg = notifier.send_incident_alert(
                severity=highest_inc["peak_severity"],
                timestamp_str=f"{highest_inc['start_str']} - {highest_inc['end_str']}",
                confidence=highest_inc["peak_confidence"]
            )
            telegram_alert_result = msg
        elif not incidents:
            telegram_alert_result = "No accident detected in video stream."

        report = {
            "success": True,
            "video_metadata": {
                "source_path": video_path,
                "duration_seconds": round(video_duration_sec, 2),
                "total_frames": total_frames,
                "analyzed_frames": analyzed_count,
                "fps": round(orig_fps, 2),
                "resolution": f"{frame_w}x{frame_h}",
                "processed_video_path": self.output_video_path
            },
            "detector_backend": detector.backend,
            "incidents_count": len(incidents),
            "incidents": incidents,
            "telegram_status": telegram_alert_result,
            "disclaimer": severity_classifier.DISCLAIMER,
            "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save structured report
        try:
            with open(self.report_json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"[VideoAnalyzer] Failed to save {self.report_json_path}: {e}")

        print(f"[VideoAnalyzer] Completed. Found {len(incidents)} distinct incident interval(s).")
        return report

    def _finalize_incident(self, inc: Dict[str, Any], incidents_list: List[Dict[str, Any]]):
        """Save evidence snapshot and append incident summary."""
        inc_id = inc["id"]
        snapshot_filename = f"{inc_id}.jpg"
        snapshot_disk_path = self.evidence_dir / snapshot_filename
        snapshot_rel_url = f"/static/evidence/{snapshot_filename}"

        if inc.get("best_frame") is not None:
            try:
                cv2.imwrite(str(snapshot_disk_path), inc["best_frame"])
            except Exception as e:
                print(f"[VideoAnalyzer] Failed to write evidence frame: {e}")

        duration = round(inc["end_sec"] - inc["start_sec"], 2)
        final_record = {
            "incident_id": inc_id,
            "start_time": inc["start_str"],
            "end_time": inc["end_str"],
            "start_sec": round(inc["start_sec"], 2),
            "end_sec": round(inc["end_sec"], 2),
            "duration_seconds": duration,
            "peak_severity": inc["peak_severity"].upper(),
            "peak_confidence": inc["peak_confidence"],
            "frame_detections": inc["frame_count"],
            "snapshot_url": snapshot_rel_url,
            "severity_details": inc.get("severity_info", {})
        }
        incidents_list.append(final_record)

# Singleton instance
video_analyzer = VideoAnalyzer()
