"""
SmartEdu - Database Seeder
Creates all tables, default login accounts, and bulk-imports the cleaned
student dataset with cached ML risk predictions.

Run: python3 seed_db.py [--limit N]
"""
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from app import create_app
from extensions import db
from models import User, Student
import ml_utils

DATA_PATH = "/home/claude/smartedu/data/clean_dataset.csv"
DEPARTMENTS = ["Computer Science", "Software Engineering", "Data Science",
               "Electrical Engineering", "Business Administration"]


def seed(limit=None):
    app = create_app()
    with app.app_context():
        db.drop_all()
        db.create_all()

        # Default accounts
        admin = User(name="Admin User", email="admin@smartedu.com", role="admin")
        admin.set_password("admin123")
        teacher = User(name="Ayesha Khan", email="teacher@smartedu.com", role="teacher")
        teacher.set_password("teacher123")
        db.session.add_all([admin, teacher])
        db.session.commit()
        print("Created default accounts: admin@smartedu.com / admin123, teacher@smartedu.com / teacher123")

        # Import students
        df = pd.read_csv(DATA_PATH)
        if limit:
            df = df.head(limit)

        rng = np.random.default_rng(42)
        model, scaler, encoder, metrics = ml_utils._load()

        # Vectorized batch prediction for speed
        X = df[["Attendance_Percentage", "Quiz_Marks", "Assignment_Marks", "Midterm_Marks"]]
        X_in = scaler.transform(X) if metrics.get("uses_scaler") else X.values
        preds = model.predict(X_in)
        probas = model.predict_proba(X_in)
        risk_labels = encoder.inverse_transform(preds)
        confidences = probas.max(axis=1) * 100

        students = []
        for i, row in df.iterrows():
            dept = DEPARTMENTS[i % len(DEPARTMENTS)]
            semester = int(rng.integers(1, 9))
            section = rng.choice(["A", "B", "C"])
            students.append(Student(
                roll_number=row["Roll_Number"],
                name=row["Student_Name"],
                father_name="N/A",
                gender=rng.choice(["Male", "Female"]),
                department=dept,
                semester=semester,
                section=section,
                session="2023-2027",
                phone="N/A",
                email=f"{row['Roll_Number'].lower()}@student.smartedu.com",
                attendance_percentage=row["Attendance_Percentage"],
                quiz_marks=row["Quiz_Marks"],
                assignment_marks=row["Assignment_Marks"],
                midterm_marks=row["Midterm_Marks"],
                final_marks=row["Final_Marks"],
                result=row["Result"],
                risk_level=risk_labels[i] if not limit else risk_labels[df.index.get_loc(i)],
                confidence=round(float(confidences[df.index.get_loc(i)]), 1),
                predicted_at=datetime.utcnow(),
            ))

        db.session.bulk_save_objects(students)
        db.session.commit()
        print(f"Imported {len(students)} students with cached ML predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of students imported")
    args = parser.parse_args()
    seed(limit=args.limit)
