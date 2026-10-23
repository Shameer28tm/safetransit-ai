
# ROAD ACCIDENT DETECTION AND PREDICTION SYSTEM

## Overview
This project is a real-time road accident detection and prediction system using:
- **YOLOv8** for accident detection from video feeds.
- **Random Forest** model for accident severity classification.
- **Flask** as the backend web framework.
- **Telegram & Sound alerts** for real-time notifications.

## Features
- Live video detection using YOLOv8 (`best.pt`)
- Accident severity prediction (Low/Medium/High) using `severity_model.pkl`
- Video upload for detection
- User login/registration (`users.json`)
- Alerts via Telegram and audio (`alert.mp3`)

## Project Structure
```
ACCIDENT DETECTION/
├── app.py                   # Main Flask application
├── best.pt                  # Trained YOLOv8 model
├── severity_model.pkl       # Trained Random Forest model
├── train_model.py           # Script to train the model
├── accident_data.csv        # Dataset used for training
├── users.json               # Login credentials data
├── templates/               # HTML templates (login, home, detect, etc.)
├── static/                  # CSS, images, audio (alert.mp3)
├── uploaded.mp4             # Sample video for testing
├── cr.mp4 / without acc.mp4 # More sample videos
└── .venv/                   # Python virtual environment
```

## Setup Instructions

1. **Create and activate virtual environment** (if not already):
   ```
   python -m venv venv
   venv\Scripts\activate (on Windows)
   source venv/bin/activate (on Linux/Mac)
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Flask app**:
   ```
   python app.py
   ```

4. **Open in browser**:
   ```
   http://127.0.0.1:5000/
   ```

## Usage
- Register/login to access the dashboard.
- Use live webcam or upload video to detect accidents.
- Severity of accidents is shown after detection.
- Alerts are triggered if an accident is detected.

## Requirements
- Python 3.8+
- Flask
- OpenCV
- Scikit-learn
- YOLOv8
- Numpy, Pandas, etc.

## Authors
- Gaurav Bharti
- M Pritham
- Md Ali
- Vinay Kumar

## Guide
- Anupama P V

## College
- Gopalan College of Engineering & Management
## for any issue 
if any problem to run this code connect me i will help you to run this project 
moblie no -9795383832
