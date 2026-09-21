import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml")

_model = None
_scaler = None
_encoder = None
_metrics = None


def _load():
    global _model, _scaler, _encoder, _metrics
    if _model is None:
        _model = joblib.load(os.path.join(ML_DIR, "risk_model.pkl"))
        _scaler = joblib.load(os.path.join(ML_DIR, "scaler.pkl"))
        _encoder = joblib.load(os.path.join(ML_DIR, "label_encoder.pkl"))
        with open(os.path.join(ML_DIR, "metrics.json")) as f:
            _metrics = json.load(f)
    return _model, _scaler, _encoder, _metrics


def get_metrics():
    _, _, _, metrics = _load()
    return metrics


def predict_risk(attendance, quiz, assignment, midterm):
    """Predict risk level for a student. Returns (risk_level, confidence_pct)."""
    model, scaler, encoder, metrics = _load()
    X = pd.DataFrame([[attendance, quiz, assignment, midterm]], columns=metrics["features"])

    if metrics.get("uses_scaler"):
        X = scaler.transform(X)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = round(float(np.max(proba)) * 100, 1)
    risk_level = encoder.inverse_transform([pred])[0]
    return risk_level, confidence


RECOMMENDATIONS = {
    "High Risk": [
        "Schedule an immediate one-on-one meeting with the teacher",
        "Assign extra practice assignments and quizzes",
        "Enable weekly progress monitoring",
        "Notify parents/guardians about academic risk",
    ],
    "Medium Risk": [
        "Encourage improved class attendance",
        "Ensure timely submission of all assignments",
        "Provide extra practice material for weak subjects",
        "Schedule a mid-semester check-in",
    ],
    "Low Risk": [
        "Maintain current performance and study habits",
        "Encourage participation in extracurricular activities",
        "Consider peer-mentoring opportunities",
    ],
}


def get_recommendations(risk_level):
    return RECOMMENDATIONS.get(risk_level, RECOMMENDATIONS["Medium Risk"])
