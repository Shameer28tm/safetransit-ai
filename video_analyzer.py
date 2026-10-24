import os
import cv2
import json
import time
import uuid
import shutil
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from yolo_detector import detector
from severity_classifier import severity_classifier
from telegram_notifier import notifier

SEVERITY_ORDER = {"fatal": 3, "major": 2, "minor": 1, "low": 1, "medium": 2, "high": 3, "unknown": 0}

class VideoAnalyzer:
    """
    Production-grade end-to-end video accident analysis engine for SafeTransit-AI.
    
    Features:
    - Per-upload job isolation (static/reports/<job_id>/) preventing cross-job data overwrites
    - File validation (container format, codec, file size)
    - Full-duration multi-frame scanning
    - User-configurable detection confidence threshold (defaults to 0.25)
    - Temporal incident coalescing (groups continuing detections within 2.5s into cohesive events)
    - Automated photographic evidence snapshot capture
    - Headless OpenCV VideoWriter web-compatible MP4 encoding (no GUI dependencies)
    - Structured JSON report export and persistence
    - Synchronization with latest incident report pointer
    """

    ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    def __init__(
        self,
        reports_base_dir: str = "static/reports",
        latest_report_json: str = "static/incident_report.json",
        latest_video_path: str = "static/processed_video.mp4",
        legacy_evidence_dir: str = "static/evidence"
    ):
        self.reports_base_dir = Path(reports_base_dir)
        self.reports_base_dir.mkdir(parents=True, exist_ok=True)
        self.latest_report_json = latest_report_json
        self.latest_video_path = latest_video_path
        self.legacy_evidence_dir = Path(legacy_evidence_dir)
        self.legacy_evidence_dir.mkdir(parents=True, exist_ok=True)

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
        conf_threshold: float = 0.25,
        target_fps: Optional[int] = 15,
        trigger_telegram: bool = True,
        progress_callback=None
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

        # Create isolated job directory
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        job_dir = self.reports_base_dir / job_id
        job_evidence_dir = job_dir / "evidence"
        job_evidence_dir.mkdir(parents=True, exist_ok=True)
        
        job_video_output = job_dir / "processed_video.mp4"
        job_report_json = job_dir / "incident_report.json"

        # Output dimensions (standardized even integers for ffmpeg/mp4v compatibility)
        # Preserve source detail; only correct odd dimensions for MP4 compatibility.
        out_w = frame_w if frame_w > 0 else 1280
        out_h = frame_h if frame_h > 0 else 720
        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1
        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1

        # Determine frame sampling step
        # If video is short (< 30s), analyze EVERY frame for maximum detection fidelity
        if video_duration_sec <= 30.0 or target_fps is None or target_fps <= 0:
            step = 1
        else:
            step = max(1, int(round(orig_fps / target_fps)))

        # Setup VideoWriter - try mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = orig_fps / step
        out = cv2.VideoWriter(str(job_video_output), fourcc, out_fps, (out_w, out_h))

        current_incident = None
        incidents = []
        frame_idx = 0
        analyzed_count = 0
        telegram_alert_result = "No accident detected in video stream."

        print(f"[VideoAnalyzer] Job {job_id}: Analyzing {video_path} ({total_frames} frames, {video_duration_sec:.1f}s, thresh={conf_threshold}, step={step})", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if step > 1 and frame_idx % step != 0:
                frame_idx += 1
                continue

            analyzed_count += 1
            if progress_callback:
                try:
                    progress_callback({"progress": min(99, int((frame_idx + 1) / max(1, total_frames) * 100)), "analyzed_frames": analyzed_count, "total_frames": total_frames, "stage": f"Analyzing frame {frame_idx + 1} of {total_frames}…"})
                except Exception:
                    pass
            timestamp_sec = frame_idx / orig_fps
            timestamp_str = f"{int(timestamp_sec // 60):02d}:{timestamp_sec % 60:05.2f}"

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
                
                badge_text = f"{d['label'].upper()} {d['confidence']*100:.1f}%"
                if is_acc and frame_severity_label != "none":
                    badge_text += f" [{frame_severity_label.upper()}]"

                # Label background badge
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1_scaled, max(0, y1_scaled - 22)), (x1_scaled + tw + 6, max(0, y1_scaled)), color, -1)
                cv2.putText(annotated_frame, badge_text, (x1_scaled + 3, max(15, y1_scaled - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw HUD
            hud_text = f"SafeTransit-AI | Time: {timestamp_str} | Threshold: {conf_threshold}"
            cv2.rectangle(annotated_frame, (10, 8), (len(hud_text) * 12 + 20, 36), (15, 23, 42), -1)
            cv2.putText(annotated_frame, hud_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1, cv2.LINE_AA)

            # Temporal incident tracking & coalescing
            if frame_has_accident:
                peak_conf = max(d['confidence'] for d in accident_dets)

                if current_incident is None:
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
                else:
                    # Check gap: if <= 2.5s, group into same incident
                    gap = timestamp_sec - current_incident["end_sec"]
                    if gap <= 2.5:
                        current_incident["end_sec"] = timestamp_sec
                        current_incident["end_str"] = timestamp_str
                        current_incident["frame_count"] += 1
                        
                        if peak_conf > current_incident["peak_confidence"]:
                            current_incident["peak_confidence"] = round(peak_conf, 4)
                            current_incident["best_frame"] = annotated_frame.copy()
                        
                        curr_rank = SEVERITY_ORDER.get(current_incident["peak_severity"], 0)
                        new_rank = SEVERITY_ORDER.get(frame_severity_label, 0)
                        if new_rank > curr_rank:
                            current_incident["peak_severity"] = frame_severity_label
                            current_incident["severity_info"] = frame_severity_info
                    else:
                        self._finalize_incident(current_incident, incidents, job_evidence_dir, job_id)
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
        if progress_callback:
            try:
                progress_callback({"progress": 99, "analyzed_frames": analyzed_count, "total_frames": total_frames, "stage": "Finalizing annotated video and incident evidence…"})
            except Exception:
                pass
        # Browser playback is most reliable with H.264/yuv420p. Re-encode if FFmpeg exists.
        try:
            import subprocess
            import shutil as _shutil
            ffmpeg = _shutil.which("ffmpeg")
            if ffmpeg and job_video_output.exists() and job_video_output.stat().st_size > 0:
                temp_h264 = job_dir / "processed_video_h264.mp4"
                proc = subprocess.run([ffmpeg, "-y", "-i", str(job_video_output), "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(temp_h264)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
                if proc.returncode == 0 and temp_h264.exists() and temp_h264.stat().st_size > 0:
                    temp_h264.replace(job_video_output)
        except Exception as encode_error:
            print(f"[VideoAnalyzer] H.264 optimization skipped: {encode_error}", flush=True)

        # Finalize any pending incident
        if current_incident is not None:
            self._finalize_incident(current_incident, incidents, job_evidence_dir, job_id)

        # Trigger Telegram alert if requested and incident exists
        if incidents and trigger_telegram:
            highest_inc = max(incidents, key=lambda x: (SEVERITY_ORDER.get(x["peak_severity"], 0), x["peak_confidence"]))
            success, msg = notifier.send_incident_alert(
                severity=highest_inc["peak_severity"],
                timestamp_str=f"{highest_inc['start_time']} - {highest_inc['end_time']}",
                confidence=highest_inc["peak_confidence"],
                force=True
            )
            telegram_alert_result = msg
        elif not incidents:
            telegram_alert_result = "No accident detected in video stream."

        # Severity breakdown distribution
        severity_dist = {"fatal": 0, "major": 0, "minor": 0}
        for inc in incidents:
            s_label = inc.get("peak_severity", "minor").lower()
            if s_label in severity_dist:
                severity_dist[s_label] += 1

        rel_job_video_path = f"/static/reports/{job_id}/processed_video.mp4"

        report = {
            "success": True,
            "job_id": job_id,
            "video_metadata": {
                "source_path": video_path,
                "duration_seconds": round(video_duration_sec, 2),
                "total_frames": total_frames,
                "analyzed_frames": analyzed_count,
                "fps": round(orig_fps, 2),
                "resolution": f"{frame_w}x{frame_h}",
                "processed_video_path": rel_job_video_path
            },
            "detector_backend": detector.backend,
            "confidence_threshold": conf_threshold,
            "incidents_count": len(incidents),
            "incidents": incidents,
            "severity_distribution": severity_dist,
            "telegram_status": telegram_alert_result,
            "disclaimer": severity_classifier.DISCLAIMER,
            "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save isolated job report
        try:
            with open(job_report_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"[VideoAnalyzer] Failed to save {job_report_json}: {e}")

        # Update latest pointer files for backward-compatibility with dashboard & existing routes
        try:
            with open(self.latest_report_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            # Copy processed video to latest pointer
            if os.path.exists(str(job_video_output)):
                shutil.copyfile(str(job_video_output), self.latest_video_path)
        except Exception as e:
            print(f"[VideoAnalyzer] Failed to update latest pointer: {e}")

        print(f"[VideoAnalyzer] Job {job_id} complete. Detected {len(incidents)} incident(s).", flush=True)
        if progress_callback:
            try:
                progress_callback({"progress": 100, "analyzed_frames": analyzed_count, "total_frames": total_frames, "stage": "Analysis complete"})
            except Exception:
                pass
        return report

    def _finalize_incident(
        self,
        inc: Dict[str, Any],
        incidents_list: List[Dict[str, Any]],
        job_evidence_dir: Path,
        job_id: str
    ):
        """Save evidence snapshot in both job evidence folder and legacy evidence folder."""
        inc_id = inc["id"]
        snapshot_filename = f"{inc_id}.jpg"
        
        job_snapshot_path = job_evidence_dir / snapshot_filename
        legacy_snapshot_path = self.legacy_evidence_dir / snapshot_filename
        
        rel_snapshot_url = f"/static/reports/{job_id}/evidence/{snapshot_filename}"

        if inc.get("best_frame") is not None:
            try:
                cv2.imwrite(str(job_snapshot_path), inc["best_frame"])
                # Also save to legacy folder for direct /static/evidence access
                cv2.imwrite(str(legacy_snapshot_path), inc["best_frame"])
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
            "snapshot_url": rel_snapshot_url,
            "severity_details": inc.get("severity_info", {})
        }
        incidents_list.append(final_record)

# Singleton instance
video_analyzer = VideoAnalyzer()
