from flask import Flask, render_template, request, redirect, session, url_for, flash, Response, jsonify
import os
import json
import time
from datetime import timedelta
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import pygame

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# SafeTransit-AI modular engines
from telegram_notifier import notifier
from severity_classifier import severity_classifier
from yolo_detector import detector
from video_analyzer import video_analyzer

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'safetransit_dev_fallback_secret_key')
app.permanent_session_lifetime = timedelta(minutes=30)

# --------------------------
# Paths and Core Settings
# --------------------------
USER_FILE = 'users.json'
ALERT_PATH = str(Path("static/alert.mp3").resolve())
VIDEO_UPLOAD_PATH = 'static/uploaded_video.mp4'
PROCESSED_VIDEO_PATH = 'static/processed_video.mp4'
REPORT_JSON_PATH = 'static/incident_report.json'

# Audio Alert Setup
pygame.mixer.init()

# Create users store if absent
if not os.path.exists(USER_FILE):
    with open(USER_FILE, 'w', encoding='utf-8') as f:
        json.dump({}, f)

def play_alert():
    """Play audible audio alert safely."""
    try:
        if os.path.exists(ALERT_PATH):
            pygame.mixer.music.load(ALERT_PATH)
            pygame.mixer.music.play()
    except Exception as e:
        print("[Audio] Notice: pygame audio playback skipped:", e)

def send_telegram_alert(severity='major'):
    """Delegate to modular TelegramNotifier with cooldown and secret masking."""
    return notifier.send_incident_alert(severity=severity)

# --------------------------
# Live Webcam Detection Generator
# --------------------------
alert_sent_live = False

def detect_accident(video_source=0):
    """
    Generator streaming processed webcam frames with accident and vehicle bounding boxes.
    If camera 0 is unavailable, yields a clean diagnostic placeholder frame.
    """
    global alert_sent_live
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"[LiveDetection] Camera index {video_source} not accessible. Streaming diagnostic HUD.")
        # Produce a synthetic diagnostic stream frame
        frame = np.zeros((500, 1020, 3), dtype=np.uint8)
        cv2.putText(frame, "SafeTransit-AI Live Camera Monitor", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 247, 255), 2)
        cv2.putText(frame, f"Video Source {video_source} disconnected or busy.", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f"Inference Backend: {detector.backend}", (50, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
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
        detections = detector.detect(frame, conf_threshold=float(os.environ.get('CONFIDENCE_THRESHOLD', 0.35)))
        
        for d in detections:
            x1, y1, x2, y2 = d['box']
            label = d['label']
            conf = d['confidence']
            is_accident = label.lower() == 'accident'
            
            # Predict severity if accident
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
                    send_telegram_alert(severity=severity_str)
                    alert_sent_live = True

            color = (0, 0, 255) if is_accident else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf*100:.0f}% [{severity_str}]", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --------------------------
# Flask Routes
# --------------------------
@app.route('/')
def root():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        with open(USER_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username in users and users[username] == password:
            session['user'] = username
            return redirect('/home')
        else:
            flash("Invalid credentials. Please verify your username and password.")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        with open(USER_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            flash("Username and password cannot be empty.")
            return render_template('register.html')
        users[username] = password
        with open(USER_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2)
        flash("Registered successfully. Please log in.")
        return redirect('/login')
    return render_template('register.html')

@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html')
    return redirect('/login')

@app.route('/live')
def live():
    if 'user' in session:
        return render_template('live.html')
    return redirect('/login')

@app.route('/video_feed')
def video_feed():
    return Response(detect_accident(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect('/login')
    if request.method == 'POST':
        if 'video' not in request.files:
            flash("No video file selected.")
            return render_template('upload.html')
        file = request.files['video']
        if file.filename == '':
            flash("Please choose a valid video file to upload.")
            return render_template('upload.html')
        
        # Verify extension
        ext = Path(file.filename).suffix.lower()
        if ext not in video_analyzer.ALLOWED_EXTENSIONS:
            flash(f"Unsupported video format '{ext}'. Supported: {', '.join(sorted(video_analyzer.ALLOWED_EXTENSIONS))}")
            return render_template('upload.html')

        try:
            file.save(VIDEO_UPLOAD_PATH)
            return redirect('/process_video')
        except Exception as e:
            flash(f"Error saving uploaded file: {e}")
            return render_template('upload.html')

    return render_template('upload.html')

@app.route('/process_video')
def process_video():
    if 'user' not in session:
        return redirect('/login')
    
    if not os.path.exists(VIDEO_UPLOAD_PATH):
        flash("No uploaded video found. Please upload a video first.")
        return redirect('/upload')

    conf_thresh = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.35))
    
    # Run full multi-frame analysis engine without GUI windows or hangs
    report = video_analyzer.analyze_video(
        video_path=VIDEO_UPLOAD_PATH,
        conf_threshold=conf_thresh,
        target_fps=15,
        trigger_telegram=True
    )

    if not report.get("success", False):
        flash(f"Analysis failed: {report.get('error', 'Unknown video processing error')}")
        return redirect('/upload')

    play_alert()
    return render_template('video_report.html', report=report)

@app.route('/test_telegram', methods=['GET', 'POST'])
def test_telegram():
    """Manual safe Telegram diagnostic test trigger."""
    if 'user' not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    success, msg = notifier.test_connection()
    return jsonify({
        "success": success,
        "message": msg,
        "bot_token_masked": notifier.get_masked_token(),
        "chat_id_masked": notifier.get_masked_chat_id(),
        "cooldown_seconds": notifier.cooldown_seconds
    })

@app.route('/api/incidents')
def api_incidents():
    """Return JSON analysis of the last processed video."""
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if os.path.exists(REPORT_JSON_PATH):
        try:
            with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "No incident report available yet."}), 404

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1')
    print(f"[SafeTransit-AI] Server starting on http://127.0.0.1:{port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
