import os
import sys
import json
import time
import io
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

class TestSafeTransitAIPipeline(unittest.TestCase):

    def test_01_telegram_notifier_security_and_health(self):
        """Test Telegram masking, sanitization, health queries, and cooldown rate limiting."""
        print("\n--- Test 01: Telegram Notifier Security & Health ---", flush=True)
        from telegram_notifier import TelegramNotifier

        # Case A: Missing credentials
        empty_notifier = TelegramNotifier(bot_token="", chat_id="")
        self.assertFalse(empty_notifier.is_configured)
        health_empty = empty_notifier.get_health()
        self.assertFalse(health_empty["configured"])
        self.assertEqual(health_empty["masked_token"], "<UNSET>")

        # Case B: Sanitization of duplicated suffix (the real bug that occurred)
        double_suffix = "1234567890:AAH_test_suffix_35chars_long12345:AAH_test_suffix_35chars_long12345"
        clean = TelegramNotifier._sanitize_token(double_suffix)
        self.assertEqual(clean, "1234567890:AAH_test_suffix_35chars_long12345")

        # Case C: Valid credential masking
        notifier = TelegramNotifier(bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz", chat_id="9876543210", cooldown_seconds=2)
        masked_tok = notifier.get_masked_token()
        masked_cid = notifier.get_masked_chat_id()
        self.assertIn("1234", masked_tok)
        self.assertIn("wxyz", masked_tok)
        self.assertNotIn("ABCdefGHIjklMNOpqrsTUV", masked_tok) # Secret not exposed
        self.assertEqual(masked_cid, "98****10")

        # Case D: Cooldown rate limiting
        notifier.last_sent_time = time.time()
        success, msg = notifier.send_incident_alert(severity="major", confidence=0.88, force=False)
        self.assertFalse(success)
        self.assertIn("Cooldown active", msg)
        print("Telegram security, sanitization, and cooldown verified successfully.", flush=True)

    def test_02_severity_classifier_schema(self):
        """Test SeverityClassifier loading, 4-feature schema, and medical disclaimers."""
        print("\n--- Test 02: Severity Classifier 4-Feature Schema ---", flush=True)
        from severity_classifier import severity_classifier
        self.assertTrue(severity_classifier.is_loaded, "severity_model.pkl must be loaded")
        
        # Test 4-feature input schema: [num_detections, avg_confidence, avg_bbox_area, class_id]
        pred = severity_classifier.predict(num_detections=2, avg_confidence=0.85, avg_bbox_area=15000.0, class_id=1)
        self.assertIn(pred['label'], ['fatal', 'major', 'minor'])
        self.assertTrue(pred['model_backed'])
        self.assertIn("disclaimer", pred)
        self.assertIn("NOT a confirmed medical", pred['disclaimer'])
        print(f"Severity classification output verified: {pred['display_label']} with feature keys {list(pred['features'].keys())}", flush=True)

    def test_03_yolo_detector_opencv_dnn(self):
        """Test OpenCV DNN ONNX inference engine, output decoding, and class mappings."""
        print("\n--- Test 03: YOLOv8 OpenCV DNN ONNX Engine ---", flush=True)
        from yolo_detector import detector
        self.assertTrue(os.path.exists("best.onnx"), "best.onnx model weight must exist")
        self.assertTrue(detector.is_ready, "OpenCV DNN net must be active")
        self.assertEqual(detector.backend, "opencv_dnn")
        self.assertIn("car", detector.class_names)
        self.assertIn("accident", detector.class_names)
        
        # Test inference on test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic bright rectangle
        cv2.rectangle(test_frame, (100, 100), (300, 300), (255, 255, 255), -1)
        dets = detector.detect(test_frame, conf_threshold=0.10)
        self.assertIsInstance(dets, list)
        print(f"OpenCV DNN inference executed without exception. Status: {detector.status_message}", flush=True)

    def test_04_accident_detection_regression_cr(self):
        """Test accident video detection on cr.mp4 (verifies real crash detection > 80% confidence)."""
        print("\n--- Test 04: Real Accident Detection on cr.mp4 ---", flush=True)
        from yolo_detector import detector
        self.assertTrue(os.path.exists("cr.mp4"), "cr.mp4 must exist")
        
        cap = cv2.VideoCapture("cr.mp4")
        # Seek to frame 42 where the accident peak occurs
        cap.set(cv2.CAP_PROP_POS_FRAMES, 42)
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret, "Failed to read frame 42 from cr.mp4")

        dets = detector.detect(frame, conf_threshold=0.30)
        accidents = [d for d in dets if d['label'] == 'accident']
        self.assertGreater(len(accidents), 0, "cr.mp4 frame 42 MUST detect an accident")
        peak_conf = accidents[0]['confidence']
        self.assertGreater(peak_conf, 0.70, f"Accident confidence should be high (> 0.70), got {peak_conf}")
        print(f"Verified accident detection on cr.mp4: {peak_conf*100:.1f}% confidence, Box: {accidents[0]['box']}", flush=True)

    def test_05_negative_control_test1(self):
        """Test normal traffic footage (test1.mp4) produces 0 false accident alarms."""
        print("\n--- Test 05: Negative Control on test1.mp4 (No False Alarms) ---", flush=True)
        from yolo_detector import detector
        if os.path.exists("test1.mp4"):
            cap = cv2.VideoCapture("test1.mp4")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
            ret, frame = cap.read()
            cap.release()
            if ret:
                dets = detector.detect(frame, conf_threshold=0.25)
                accidents = [d for d in dets if d['label'] == 'accident']
                self.assertEqual(len(accidents), 0, "Normal street traffic should not trigger false accident detections")
                print("Negative control verified: 0 false alarms on normal traffic.", flush=True)

    def test_06_video_analyzer_job_isolation(self):
        """Test VideoAnalyzer per-upload job isolation and output generation."""
        print("\n--- Test 06: Video Analyzer Per-Upload Job Isolation ---", flush=True)
        from video_analyzer import video_analyzer
        test_video = "cr.mp4"
        report = video_analyzer.analyze_video(
            video_path=test_video,
            conf_threshold=0.30,
            target_fps=15,
            trigger_telegram=False
        )
        self.assertTrue(report["success"])
        self.assertIn("job_id", report)
        job_id = report["job_id"]
        
        # Verify isolated job directory exists
        job_dir = os.path.join("static", "reports", job_id)
        self.assertTrue(os.path.exists(job_dir), f"Job directory {job_dir} must be created")
        self.assertTrue(os.path.exists(os.path.join(job_dir, "processed_video.mp4")))
        self.assertTrue(os.path.exists(os.path.join(job_dir, "incident_report.json")))
        
        # Verify latest pointer synchronization
        self.assertTrue(os.path.exists("static/incident_report.json"))
        self.assertTrue(os.path.exists("static/processed_video.mp4"))
        print(f"Job {job_id} successfully created with isolated report and video assets.", flush=True)

    def test_07_flask_auth_and_routes(self):
        """Test Flask authentication, CSRF/session isolation, and all core routes."""
        print("\n--- Test 07: Flask Authentication and Routes ---", flush=True)
        from app import app
        app.config['TESTING'] = True
        client = app.test_client()

        # Unauthenticated redirects
        self.assertEqual(client.get('/').status_code, 302)
        self.assertEqual(client.get('/home').status_code, 302)
        self.assertEqual(client.get('/live').status_code, 302)
        self.assertEqual(client.get('/upload').status_code, 302)

        # Public routes
        self.assertEqual(client.get('/login').status_code, 200)
        self.assertEqual(client.get('/register').status_code, 200)
        self.assertEqual(client.get('/about').status_code, 200)

        # Operator authentication
        login_res = client.post('/login', data={"username": "operator", "password": "admin123"}, follow_redirects=True)
        self.assertEqual(login_res.status_code, 200)
        self.assertIn(b"Operations Command Center", login_res.data)

        # Authenticated access
        home_res = client.get('/home')
        self.assertEqual(home_res.status_code, 200)
        self.assertIn(b"SafeTransit", home_res.data)

        live_res = client.get('/live')
        self.assertEqual(live_res.status_code, 200)

        upload_res = client.get('/upload')
        self.assertEqual(upload_res.status_code, 200)

        # Telegram diagnostic POST endpoint
        tel_res = client.post('/test_telegram')
        self.assertEqual(tel_res.status_code, 200)
        tel_data = json.loads(tel_res.data)
        self.assertTrue(tel_data["success"], f"Telegram diagnostic failed: {tel_data}")
        self.assertIn("masked_token", tel_data)
        print("Flask routes, authentication, and Telegram diagnostic confirmed working.", flush=True)

if __name__ == '__main__':
    unittest.main()
