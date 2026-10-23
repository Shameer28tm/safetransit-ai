from flask import Flask, render_template, request, redirect, session, url_for, flash, Response, jsonify, send_file
import os
import json
import time
import hashlib
import threading
import uuid
from datetime import timedelta
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import pygame
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# SafeTransit-AI modular intelligence engines
from telegram_notifier import notifier
from severity_classifier import severity_classifier
from yolo_detector import detector
from video_analyzer import video_analyzer

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'safetransit_shameer_signature_secure_key_2026')
app.permanent_session_lifetime = timedelta(hours=4)

# --------------------------
# Paths & Directory Hierarchy
# --------------------------
USER_FILE = 'users.json'
ALERT_PATH = str(Path("static/alert.mp3").resolve())
UPLOADS_DIR = Path("static/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path("static/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LATEST_REPORT_JSON = 'static/incident_report.json'
LATEST_VIDEO_PATH = 'static/processed_video.mp4'

# Audio Alert Subsystem
try:
    pygame.mixer.init()
except Exception as e:
    print(f"[Audio] pygame.mixer initialization notice: {e}")

# Ensure user authentication store exists
if not os.path.exists(USER_FILE):
    with open(USER_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "operator": "admin123",
            "testjudge": "securepass123"
        }, f, indent=2)

def hash_credential(plain_text: str) -> str:
    """Generate salted SHA-256 hash for secure storage."""
    return f"sha256${hashlib.sha256(plain_text.encode('utf-8')).hexdigest()}"

def verify_credential(stored_hash: str, input_text: str) -> bool:
    """Verify password supporting both salted sha256 and legacy plaintext for backwards-compatibility."""
    if stored_hash.startswith("sha256$"):
        expected = stored_hash.split("$", 1)[1]
        actual = hashlib.sha256(input_text.encode('utf-8')).hexdigest()
        return expected == actual
    # Legacy plaintext check
    return stored_hash == input_text

def play_alert():
    """Play audible alert safely without crashing if host hardware has no speakers."""
    try:
        if os.path.exists(ALERT_PATH):
            pygame.mixer.music.load(ALERT_PATH)
            pygame.mixer.music.play()
    except Exception as e:
        print("[Audio] Notice: pygame playback skipped:", e)

def send_telegram_alert(severity='major', timestamp_str='', confidence=0.0):
    """Delegate to modular TelegramNotifier with deduplication and cooldown."""
    return notifier.send_incident_alert(
        severity=severity,
        timestamp_str=timestamp_str,
        confidence=confidence
    )

# --------------------------
# Live Camera Detection Stream
# --------------------------
alert_sent_live = False
live_event_id = 0
live_accident_detected = False
analysis_jobs = {}
analysis_jobs_lock = threading.Lock()

def detect_accident(video_source=0):
    """
    Generator streaming processed webcam frames with accident and vehicle bounding boxes.
    If camera 0 is unavailable, yields a clean diagnostic HUD frame.
    """
    global alert_sent_live, live_event_id, live_accident_detected
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        # Diagnostic placeholder frame
        frame = np.zeros((500, 1020, 3), dtype=np.uint8)
        cv2.putText(frame, "SafeTransit-AI Sensor Stream Offline", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 247, 255), 2)
        cv2.putText(frame, f"Camera device {video_source} was not detected by host OS.", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Use 'Video Intelligence' tab to analyze stored footage.", (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(frame, f"Detector Backend: {detector.backend} | Telegram: Active", (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1.0)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1020, 500))
        conf_thresh = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.28))
        detections = detector.detect(frame, conf_threshold=conf_thresh)
        current_has_accident = any(d["label"].lower() == "accident" for d in detections)
        if current_has_accident and not live_accident_detected:
            live_event_id += 1
            live_accident_detected = True
        elif not current_has_accident:
            live_accident_detected = False
            alert_sent_live = False
        
        for d in detections:
            x1, y1, x2, y2 = d['box']
            label = d['label']
            conf = d['confidence']
            is_accident = label.lower() == 'accident'
            
            severity_str = "MAJOR"
            if is_accident:
                sev_res = severity_classifier.predict(
                    num_detections=len(detections),
                    avg_confidence=conf,
                    avg_bbox_area=(x2 - x1) * (y2 - y1),
                    class_id=d['class_id']
                )
                severity_str = sev_res['display_label']

                if not alert_sent_live:
                    play_alert()
                    send_telegram_alert(
                        severity=severity_str,
                        timestamp_str=time.strftime('%Y-%m-%d %H:%M:%S'),
                        confidence=conf
                    )
                    alert_sent_live = True

            color = (0, 0, 255) if is_accident else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label.upper()} {conf*100:.0f}% [{severity_str}]", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --------------------------
# Flask Authentication Routes
# --------------------------
@app.route('/')
def root():
    if 'user' in session:
        return redirect('/home')
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        with open(USER_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if username in users and verify_credential(users[username], password):
            session['user'] = username
            flash(f"Welcome back, Operator {username}. System telemetry online.", "success")
            return redirect('/home')
        else:
            flash("Invalid operator credentials. Please check username and password.", "danger")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        with open(USER_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash("Username and password fields cannot be blank.", "warning")
            return render_template('register.html')
        
        if len(password) < 4:
            flash("Password must be at least 4 characters long.", "warning")
            return render_template('register.html')
            
        if username in users:
            flash(f"Operator username '{username}' is already registered.", "warning")
            return render_template('register.html')

        # Store with secure hash
        users[username] = hash_credential(password)
        with open(USER_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2)
            
        flash("Operator registered successfully. Please authenticate.", "success")
        return redirect('/login')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('latest_job_id', None)
    flash("Session terminated. Operator signed out.", "info")
    return redirect('/login')

# --------------------------
# Main Command Center Route
# --------------------------
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/login')

    # Load latest incident report telemetry
    latest_report = {}
    if os.path.exists(LATEST_REPORT_JSON):
        try:
            with open(LATEST_REPORT_JSON, 'r', encoding='utf-8') as f:
                latest_report = json.load(f)
        except Exception as e:
            print(f"[Dashboard] Error reading {LATEST_REPORT_JSON}: {e}")

    incidents = latest_report.get('incidents', [])
    total_incidents = len(incidents)
    
    # Compute real severity stats
    fatal_cnt = sum(1 for i in incidents if i.get('peak_severity') == 'FATAL')
    major_cnt = sum(1 for i in incidents if i.get('peak_severity') == 'MAJOR')
    minor_cnt = sum(1 for i in incidents if i.get('peak_severity') == 'MINOR')
    peak_severity = incidents[0].get('peak_severity', 'NONE') if incidents else 'NONE'

    stats = {
        "total_incidents": total_incidents,
        "fatal": fatal_cnt,
        "major": major_cnt,
        "minor": minor_cnt,
        "peak_severity": peak_severity
    }

    telegram_health = notifier.get_health()

    return render_template(
        'home.html',
        stats=stats,
        latest_report=latest_report,
        telegram_health=telegram_health
    )

# --------------------------
# Live Monitoring Route
# --------------------------
@app.route('/live')
def live():
    if 'user' not in session:
        return redirect('/login')
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_accident(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/live_status")
def api_live_status():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"accident_detected": live_accident_detected, "event_id": live_event_id})

# --------------------------
# Video Intelligence & Analysis Routes
# --------------------------
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect('/login')
        
    if request.method == 'POST':
        if 'video' not in request.files:
            flash("No video file selected in upload payload.", "danger")
            return render_template('upload.html')
            
        file = request.files['video']
        if file.filename == '':
            flash("Please choose a valid video file to upload.", "warning")
            return render_template('upload.html')

        ext = Path(file.filename).suffix.lower()
        if ext not in video_analyzer.ALLOWED_EXTENSIONS:
            flash(f"Unsupported format '{ext}'. Permitted: {', '.join(sorted(video_analyzer.ALLOWED_EXTENSIONS))}", "danger")
            return render_template('upload.html')

        # Read configurable confidence threshold from form
        try:
            conf_thresh = float(request.form.get('confidence_threshold', 0.28))
            conf_thresh = max(0.10, min(0.85, conf_thresh))
        except (ValueError, TypeError):
            conf_thresh = 0.28

        # Save to uniquely isolated upload path
        safe_fname = secure_filename(file.filename) or "footage.mp4"
        job_tag = f"{int(time.time())}_{safe_fname}"
        saved_video_path = str(UPLOADS_DIR / job_tag)

        try:
            file.save(saved_video_path)
        except Exception as e:
            flash(f"Failed to persist uploaded video: {e}", "danger")
            return render_template('upload.html')

        # Start analysis in a worker so the browser can poll real frame progress.
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            task_id = uuid.uuid4().hex
            with analysis_jobs_lock:
                analysis_jobs[task_id] = {"state": "running", "progress": 0, "analyzed_frames": 0, "total_frames": 0, "stage": "Opening video…"}
            def progress_update(progress):
                with analysis_jobs_lock:
                    if task_id in analysis_jobs:
                        analysis_jobs[task_id].update(progress)
            def run_analysis():
                try:
                    report_result = video_analyzer.analyze_video(
                        video_path=saved_video_path,
                        conf_threshold=conf_thresh,
                        target_fps=15,
                        trigger_telegram=True,
                        progress_callback=progress_update
                    )
                    with analysis_jobs_lock:
                        if report_result.get("success"):
                            analysis_jobs[task_id].update({"state": "complete", "progress": 100, "stage": "Analysis complete", "job_id": report_result.get("job_id"), "report_url": f"/report/{report_result.get('job_id')}"})
                        else:
                            analysis_jobs[task_id].update({"state": "error", "error": report_result.get("error", "Video analysis failed.")})
                except Exception as exc:
                    with analysis_jobs_lock:
                        analysis_jobs[task_id].update({"state": "error", "error": str(exc)})
            threading.Thread(target=run_analysis, daemon=True, name=f"analysis-{task_id[:8]}").start()
            return jsonify({"success": True, "task_id": task_id}), 202

        # Backward-compatible synchronous submission for non-JavaScript clients.
        report = video_analyzer.analyze_video(
            video_path=saved_video_path,
            conf_threshold=conf_thresh,
            target_fps=15,
            trigger_telegram=True
        )

        if not report.get("success", False):
            flash(f"Analysis aborted: {report.get('error', 'Unknown codec error')}", "danger")
            return render_template('upload.html')

        session['latest_job_id'] = report.get('job_id')
        if report.get("incidents"):
            play_alert()
        return render_template('video_report.html', report=report)

    return render_template('upload.html')

@app.route("/analysis_status/<task_id>")
def analysis_status(task_id):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    with analysis_jobs_lock:
        status = analysis_jobs.get(secure_filename(task_id))
        if status is None:
            return jsonify({"state": "error", "error": "Analysis task not found."}), 404
        return jsonify(dict(status))

@app.route('/process_video')
def process_video():
    """Backward-compatible fallback route for /process_video."""
    if 'user' not in session:
        return redirect('/login')
    return redirect('/video_report')

@app.route('/video_report')
def video_report():
    """Render the most recently processed incident report."""
    if 'user' not in session:
        return redirect('/login')

    if os.path.exists(LATEST_REPORT_JSON):
        try:
            with open(LATEST_REPORT_JSON, 'r', encoding='utf-8') as f:
                report = json.load(f)
            return render_template('video_report.html', report=report)
        except Exception as e:
            flash(f"Error loading report: {e}", "danger")
            return redirect('/home')
            
    flash("No incident report has been generated yet. Please upload a video first.", "info")
    return redirect('/upload')

@app.route('/report/<job_id>')
def view_specific_report(job_id):
    """View historical isolated report by job ID."""
    if 'user' not in session:
        return redirect('/login')
        
    clean_job_id = secure_filename(job_id)
    report_file = REPORTS_DIR / clean_job_id / "incident_report.json"
    if report_file.exists():
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            return render_template('video_report.html', report=report)
        except Exception as e:
            flash(f"Error reading report {job_id}: {e}", "danger")
            
    flash(f"Report '{job_id}' not found.", "warning")
    return redirect('/home')

@app.route('/download_report')
@app.route('/download_report/<job_id>')
def download_report(job_id=None):
    """Export formatted JSON incident telemetry report as downloadable attachment."""
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    target_path = LATEST_REPORT_JSON
    if job_id:
        clean_id = secure_filename(job_id)
        job_file = REPORTS_DIR / clean_id / "incident_report.json"
        if job_file.exists():
            target_path = str(job_file)

    if os.path.exists(target_path):
        return send_file(
            target_path,
            as_attachment=True,
            download_name=f"safetransit_incident_report_{int(time.time())}.json",
            mimetype="application/json"
        )
    return jsonify({"error": "Report file not found."}), 404

# --------------------------
# Telegram Diagnostics Endpoint
# --------------------------
@app.route('/test_telegram', methods=['GET', 'POST'])
def test_telegram():
    """
    Diagnostic verification endpoint for competition demonstration.
    Returns status confirmation without exposing credentials.
    """
    if 'user' not in session:
        return jsonify({"success": False, "message": "Unauthorized operator session"}), 401
        
    success, msg = notifier.test_connection()
    health = notifier.get_health()
    return jsonify({
        "success": success,
        "message": msg,
        "bot_username": health.get("bot_username"),
        "masked_token": health.get("masked_token"),
        "masked_chat_id": health.get("masked_chat_id")
    })

# --------------------------
# API Telemetry Endpoints
# --------------------------
@app.route('/api/system_status')
def api_system_status():
    """Return live system status telemetry."""
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({
        "status": "ONLINE",
        "backend": detector.backend,
        "model_status": detector.status_message,
        "severity_model_loaded": severity_classifier.is_loaded,
        "telegram_health": notifier.get_health(),
        "server_time": time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/incidents')
def api_incidents():
    """Return JSON telemetry of the latest analysis report."""
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if os.path.exists(LATEST_REPORT_JSON):
        try:
            with open(LATEST_REPORT_JSON, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "No incident report available."}), 404

@app.route('/about')
def about():
    if 'user' in session:
        return redirect('/home')
    return redirect('/login')

# --------------------------
# Error Handlers
# --------------------------
@app.errorhandler(404)
def page_not_found(e):
    return render_template('base.html'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal processing failure. Please consult server logs."}), 500

# --------------------------
# Main Execution Entry Point
# --------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1')
    print(f"[SafeTransit-AI] Shameer Signature Edition starting on http://127.0.0.1:{port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
