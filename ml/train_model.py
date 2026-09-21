"""
SmartEdu - ML Model Training
Trains Logistic Regression, Random Forest and XGBoost classifiers to predict
student Risk_Level (Low / Medium / High) and picks the best model based on
F1 score (macro).
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import json

DATA_PATH = "/home/claude/smartedu/data/clean_dataset.csv"
MODEL_PATH = "/home/claude/smartedu/ml/risk_model.pkl"
SCALER_PATH = "/home/claude/smartedu/ml/scaler.pkl"
ENCODER_PATH = "/home/claude/smartedu/ml/label_encoder.pkl"
METRICS_PATH = "/home/claude/smartedu/ml/metrics.json"

FEATURES = [
    "Attendance_Percentage",
    "Quiz_Marks",
    "Assignment_Marks",
    "Midterm_Marks",
]


def train():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["Risk_Level"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42, n_jobs=-1
        ),
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")
        f1 = f1_score(y_test, preds, average="macro")
        cm = confusion_matrix(y_test, preds).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
        }

        print(f"{name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({
            "best_model": best_name,
            "uses_scaler": best_name == "Logistic Regression",
            "classes": le.classes_.tolist(),
            "features": FEATURES,
            "results": results,
        }, f, indent=2)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
