from flask import Flask, render_template, request, redirect, session, url_for, flash, Response
import os, json
from datetime import timedelta
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from pathlib import Path
import pygame
import requests   # for Telegram alert
import joblib     # for loading RandomForest model
import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'safetransit_dev_fallback_secret_key')
app.permanent_session_lifetime = timedelta(minutes=30)

# --------------------------
# Path & File Setup
# --------------------------
USER_FILE = 'users.json'
MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', 'best.pt')
ALERT_PATH = str(Path("static/alert.mp3").resolve())
VIDEO_UPLOAD_PATH = 'static/uploaded_video.mp4'
PROCESSED_VIDEO_PATH = 'static/processed_video.mp4'

# --------------------------
# Telegram Setup (Loaded securely from environment variables)
# --------------------------
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# --------------------------
# Load Models and Classes
# --------------------------
model = YOLO(MODEL_PATH)

# Read class names from a text file (one per line)
with open("coco1.txt", "r") as f:
    class_list = f.read().splitlines()

# Load RandomForest Model and LabelEncoder
with open("severity_model.pkl", "rb") as f:
    rf_model, le = joblib.load(f)

# --------------------------
# Initialize Audio Alert
# --------------------------
alert_sent = False
pygame.mixer.init()

# Create users file if it does not exist
if not os.path.exists(USER_FILE):
    with open(USER_FILE, 'w') as f:
        json.dump({}, f)

# --------------------------
# Helper Functions
# --------------------------
def play_alert():
    """Play audio alert using pygame."""
    try:
        pygame.mixer.music.load(ALERT_PATH)
        pygame.mixer.music.play()
    except Exception as e:
        print("Failed to play alert:", e)

def send_telegram_alert(severity='unknown'):
    """Send Telegram alert with a message indicating accident severity."""
    if not BOT_TOKEN or not CHAT_ID or 'your_' in BOT_TOKEN or 'placeholder' in BOT_TOKEN:
        print("[Notifier] Telegram alert skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured.")
        return
    try:
        msg = f"🚨 Accident Detected!\nSeverity: {severity.upper()}"
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={msg}"
        requests.get(url, timeout=5)
    except Exception as e:
        print("Telegram alert failed:", e)

def predict_severity(features):
    """
    Predict the severity of the accident using the trained Random Forest model.
    Features should be a list of four values: num_detections, avg_confidence, avg_bbox_area, class_id.
    """
    features = np.array(features).reshape(1, -1)  # Reshape for model prediction
    severity_encoded = rf_model.predict(features)[0]
    severity = le.inverse_transform([severity_encoded])[0]
    return severity

def detect_accident(video_source=0):
    """
    Generator function for streaming video frames with accident detection.
    By default, uses the primary camera (video_source=0).
    """
    global alert_sent
    cap = cv2.VideoCapture(video_source)
    
    # Check if video source is accessible
    if not cap.isOpened():
        print("Error: Cannot access video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        boxes = results[0].boxes.data  # Prediction boxes
        px = pd.DataFrame(boxes).astype("float")

        for index, row in px.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            class_id = int(row[5])
            label = class_list[class_id]
            
            # Predict severity using the Random Forest model
            num_detections = len(px)
            avg_confidence = row[4]  # Assuming confidence is at column index 4
            avg_bbox_area = (x2 - x1) * (y2 - y1)  # Simple bounding box area calculation
            features = [num_detections, avg_confidence, avg_bbox_area, class_id]
            severity = predict_severity(features)
            
            if label.lower() == 'accident' and not alert_sent:
                play_alert()
                send_telegram_alert(severity)
                alert_sent = True
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cvzone.putTextRect(frame, f'{label} - {severity}', (x1, y1), 1, 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --------------------------
# Routes & Views
# --------------------------
@app.route('/')
def root():
    # Redirect to the login page
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        with open(USER_FILE, 'r') as f:
            users = json.load(f)
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['user'] = username
            return redirect('/home')
        else:
            flash("Invalid credentials")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        with open(USER_FILE, 'r') as f:
            users = json.load(f)
        username = request.form.get('username')
        password = request.form.get('password')
        # Ideally, passwords should be hashed instead of stored in plain text.
        users[username] = password
        with open(USER_FILE, 'w') as f:
            json.dump(users, f)
        flash("Registered successfully.")
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
    # Stream the processed video frames with accident detection.
    return Response(detect_accident(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect('/login')
    if request.method == 'POST':
        file = request.files['video']
        if file:
            file.save(VIDEO_UPLOAD_PATH)
            return redirect('/process_video')
    return render_template('upload.html')

@app.route('/process_video')
def process_video():
    cap = cv2.VideoCapture(VIDEO_UPLOAD_PATH)
    if not cap.isOpened():
        return "Error: Cannot access uploaded video."
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(PROCESSED_VIDEO_PATH, fourcc, 20.0, (1020, 500))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")

        for index, row in px.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            class_id = int(row[5])
            label = class_list[class_id]
            
            # Predict severity using the Random Forest model
            num_detections = len(px)
            avg_confidence = row[4]  # Assuming confidence is at column index 4
            avg_bbox_area = (x2 - x1) * (y2 - y1)  # Simple bounding box area calculation
            features = [num_detections, avg_confidence, avg_bbox_area, class_id]
            severity = predict_severity(features)
            
            if label.lower() == 'accident':
                play_alert()
                send_telegram_alert(severity)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cvzone.putTextRect(frame, f'{label} - {severity}', (x1, y1), 1, 1)

        out.write(frame)
        # Optionally, show the frame during processing. Press 'q' to stop if needed.
        cv2.imshow("Processing Video...", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Return a simple HTML page to show the processed video.
    return f'''
    <html>
      <body style="background-color: black; text-align: center; color: white;">
        <h2>Processed Video</h2>
        <video width="800" controls>
          <source src="/static/processed_video.mp4" type="video/mp4">
        </video><br>
        <a href="/home"><button>Home</button></a>
      </body>
    </html>
    '''

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

# --------------------------
# Main Entry Point
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
