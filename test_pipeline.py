import os
import sys
import json
import time
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

class TestSafeTransitAI(unittest.TestCase):

    def test_01_telegram_notifier(self):
        print("\n--- Running Telegram Notifier Tests ---")
        from telegram_notifier import TelegramNotifier
        # Test with dummy secrets
        notifier = TelegramNotifier(bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz", chat_id="987654321", cooldown_seconds=2)
        
        # Test masking
        masked_tok = notifier.get_masked_token()
        masked_cid = notifier.get_masked_chat_id()
        self.assertIn("1234", masked_tok)
        self.assertIn("wxyz", masked_tok)
        self.assertNotIn("ABCdefGHIjklMNOpqrsTUV", masked_tok) # Secret not exposed
        self.assertEqual(masked_cid, "98...21")
        print(f"Masked token: {masked_tok}, Masked chat ID: {masked_cid}")

        # Test cooldown logic
        notifier.last_sent_time = time.time()
        success, msg = notifier.send_incident_alert(severity="major", confidence=0.88, force=False)
        self.assertFalse(success)
        self.assertIn("Cooldown active", msg)
        print("Cooldown suppression verified:", msg)

    def test_02_severity_classifier(self):
        print("\n--- Running Severity Classifier Tests ---")
        from severity_classifier import severity_classifier
        self.assertTrue(severity_classifier.is_loaded, "severity_model.pkl should be loaded")
        
        # Test prediction schema
        pred = severity_classifier.predict(num_detections=2, avg_confidence=0.85, avg_bbox_area=15000.0, class_id=1)
        self.assertIn(pred['label'], ['fatal', 'major', 'minor'])
        self.assertTrue(pred['model_backed'])
        self.assertIn("disclaimer", pred)
        self.assertIn("NOT a confirmed medical", pred['disclaimer'])
        print(f"Severity Prediction: {pred['label'].upper()} (probabilities: {pred.get('probabilities')})")

    def test_03_yolo_detector_onnx(self):
        print("\n--- Running YOLO Detector OpenCV DNN Tests ---")
        from yolo_detector import detector
        self.assertTrue(os.path.exists("best.onnx"), "best.onnx must exist")
        self.assertTrue(detector.is_ready, "detector must be ready")
        self.assertEqual(detector.backend, "opencv_dnn")
        
        # Test inference on blank test frame
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(dummy)
        self.assertIsInstance(dets, list)
        print("OpenCV DNN backend active:", detector.status_message)

    def test_04_video_analyzer_pipeline(self):
        print("\n--- Running Video Analyzer Pipeline Test ---")
        from video_analyzer import video_analyzer
        test_video = "demo_media/highway_collision_demo.webm"
        if not os.path.exists(test_video):
            test_video = "cr.mp4"
        
        self.assertTrue(os.path.exists(test_video), f"Test video {test_video} must exist")
        report = video_analyzer.analyze_video(
            video_path=test_video,
            conf_threshold=0.35,
            target_fps=10,
            trigger_telegram=False # Safe during unit test
        )
        self.assertTrue(report["success"])
        self.assertIn("video_metadata", report)
        self.assertIn("incidents", report)
        self.assertTrue(os.path.exists("static/processed_video.mp4"))
        print(f"Video analysis completed successfully. Incidents found: {report['incidents_count']}")
        print(f"Duration: {report['video_metadata']['duration_seconds']}s, Frames analyzed: {report['video_metadata']['analyzed_frames']}")

    def test_05_flask_routes(self):
        print("\n--- Running Flask Routes Integration Tests ---")
        from app import app
        app.config['TESTING'] = True
        client = app.test_client()

        # Route /
        res = client.get('/')
        self.assertEqual(res.status_code, 302) # Redirect to login

        # Route /login GET
        res = client.get('/login')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Login", res.data)

        # Route /register GET
        res = client.get('/register')
        self.assertEqual(res.status_code, 200)

        # Route /about GET
        res = client.get('/about')
        self.assertEqual(res.status_code, 200)

        # Register and Login session
        client.post('/register', data={"username": "testjudge", "password": "securepass123"})
        login_res = client.post('/login', data={"username": "testjudge", "password": "securepass123"}, follow_redirects=True)
        self.assertEqual(login_res.status_code, 200)
        self.assertIn(b"SafeTransit-AI Command Hub", login_res.data)

        # Test Telegram route
        tel_res = client.get('/test_telegram')
        self.assertEqual(tel_res.status_code, 200)
        tel_data = json.loads(tel_res.data)
        self.assertIn("bot_token_masked", tel_data)
        print("Flask routes verified. Telegram test response:", tel_data)

if __name__ == '__main__':
    unittest.main()
