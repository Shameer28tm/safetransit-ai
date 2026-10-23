import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

class SeverityClassifier:
    """
    Severity Classifier wrapper for SafeTransit-AI.
    Features:
    - Safely loads severity_model.pkl preserving the verified (rf_model, label_encoder) tuple structure.
    - Uses the exact 4-feature schema: ['num_detections', 'avg_confidence', 'avg_bbox_area', 'class_id']
    - Handles actual classes: 'fatal', 'major', 'minor'
    - Returns structured prediction dictionary with prominent non-medical disclaimer.
    """

    FEATURE_NAMES = ['num_detections', 'avg_confidence', 'avg_bbox_area', 'class_id']
    DISCLAIMER = (
        "Estimated scene severity rating derived from computer vision metrics "
        "(vehicle count, confidence, collision bounding area). "
        "This is an operational situational indicator, NOT a confirmed medical or clinical injury diagnosis."
    )

    def __init__(self, model_path: str = "severity_model.pkl"):
        self.model_path = model_path
        self.rf_model = None
        self.label_encoder = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[SeverityClassifier] Warning: Model file {self.model_path} not found.")
            return

        try:
            with open(self.model_path, "rb") as f:
                loaded = joblib.load(f)
            
            if isinstance(loaded, (tuple, list)) and len(loaded) >= 2:
                self.rf_model = loaded[0]
                self.label_encoder = loaded[1]
                self.is_loaded = True
                print(f"[SeverityClassifier] Successfully loaded {self.model_path} with classes: {getattr(self.label_encoder, 'classes_', 'unknown')}")
            else:
                print(f"[SeverityClassifier] Warning: Expected (model, label_encoder) tuple, got {type(loaded)}.")
        except Exception as e:
            print(f"[SeverityClassifier] Error loading {self.model_path}: {e}")

    def predict(
        self,
        num_detections: int,
        avg_confidence: float,
        avg_bbox_area: float,
        class_id: int = 1
    ) -> Dict[str, Any]:
        """
        Predict scene severity from detection features.
        
        Args:
            num_detections: Number of detected accident / vehicle objects in scene
            avg_confidence: Average detection confidence score (0.0 to 1.0)
            avg_bbox_area: Average bounding box area (in pixels)
            class_id: Class ID (default 1 for 'accident')
            
        Returns:
            Dict containing label ('fatal', 'major', 'minor'), display_badge, score, and disclaimer.
        """
        # Ensure values are within reasonable numerical bounds
        num_detections = max(1, int(num_detections))
        avg_confidence = float(np.clip(avg_confidence, 0.0, 1.0))
        avg_bbox_area = max(1.0, float(avg_bbox_area))
        class_id = int(class_id)

        # Fallback heuristic if model is not loaded
        if not self.is_loaded:
            if avg_bbox_area > 40000 or (num_detections >= 3 and avg_confidence > 0.7):
                label = "fatal"
            elif avg_bbox_area > 15000 or num_detections >= 2:
                label = "major"
            else:
                label = "minor"
            return {
                "label": label,
                "display_label": label.upper(),
                "model_backed": False,
                "disclaimer": self.DISCLAIMER,
                "features": {
                    "num_detections": num_detections,
                    "avg_confidence": round(avg_confidence, 3),
                    "avg_bbox_area": round(avg_bbox_area, 1),
                    "class_id": class_id
                }
            }

        try:
            # Construct DataFrame with exact feature names expected by the scikit-learn model
            df = pd.DataFrame([[
                num_detections,
                avg_confidence,
                avg_bbox_area,
                class_id
            ]], columns=self.FEATURE_NAMES)

            pred_encoded = self.rf_model.predict(df)[0]
            label = str(self.label_encoder.inverse_transform([pred_encoded])[0]).lower()

            # Predict probabilities if available
            probabilities = {}
            if hasattr(self.rf_model, "predict_proba"):
                probs = self.rf_model.predict_proba(df)[0]
                for idx, cls_name in enumerate(self.label_encoder.classes_):
                    probabilities[str(cls_name)] = round(float(probs[idx]), 3)

            return {
                "label": label,
                "display_label": label.upper(),
                "model_backed": True,
                "probabilities": probabilities,
                "disclaimer": self.DISCLAIMER,
                "features": {
                    "num_detections": num_detections,
                    "avg_confidence": round(avg_confidence, 3),
                    "avg_bbox_area": round(avg_bbox_area, 1),
                    "class_id": class_id
                }
            }
        except Exception as e:
            print(f"[SeverityClassifier] Prediction error: {e}. Falling back to default 'major'.")
            return {
                "label": "major",
                "display_label": "MAJOR",
                "model_backed": False,
                "error": str(e),
                "disclaimer": self.DISCLAIMER,
                "features": {
                    "num_detections": num_detections,
                    "avg_confidence": round(avg_confidence, 3),
                    "avg_bbox_area": round(avg_bbox_area, 1),
                    "class_id": class_id
                }
            }

# Singleton instance
severity_classifier = SeverityClassifier()
