# SmartEdu — Student Performance Prediction System

A Flask + Machine Learning web app that predicts a student's academic risk
level (Low / Medium / High) from attendance, quiz, assignment and midterm
data, so teachers can intervene before final exams.

Built from the SmartEdu PRD, using the provided `smartedu_unclean_dataset.csv`
(102,007 raw rows → 100,009 cleaned rows after removing duplicates, invalid
values, and inconsistent labels).

## What's included

| PRD Module | Status |
|---|---|
| Authentication (login/logout/roles) | ✅ Teacher & Admin roles, hashed passwords |
| Dashboard (cards + charts) | ✅ Stat cards, risk/attendance/marks charts (Chart.js) |
| Student Management | ✅ Add / Edit / Delete / Search / Filter / Pagination |
| Attendance Management | ✅ Daily entry, auto % recalculation |
| Marks Management | ✅ Quiz / Assignment / Midterm / Final, auto totals |
| ML Prediction | ✅ Logistic Regression / Random Forest / XGBoost — best model auto-selected |
| Recommendation Engine | ✅ Risk-based suggestions per student |
| Reports | ✅ CSV export (all/by risk), individual PDF report |
| Analytics Dashboard | ✅ Doughnut / bar / radar charts |
| Notifications | ✅ In-app high-risk & low-attendance alert panels |
| Admin Panel | ✅ Manage teacher/admin accounts |

Not included in this first build (flagged as **Future Scope** in the PRD
itself): Student/Parent portals, mobile app, face-recognition attendance.
These can be added incrementally.

## 1. Setup

```bash
cd smartedu
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 2. Clean the data & train the ML model (already done once, re-run if you replace the CSV)

```bash
python3 ml/clean_data.py      # -> data/clean_dataset.csv
python3 ml/train_model.py     # -> ml/risk_model.pkl, scaler.pkl, label_encoder.pkl, metrics.json
```

Model comparison on the held-out test set (20%):

| Model | Accuracy | F1 (macro) |
|---|---|---|
| **Logistic Regression (selected)** | **95.0%** | **93.3%** |
| Random Forest | 94.6% | 93.0% |
| XGBoost | 94.8% | 93.1% |

All three clear the PRD's 80% accuracy target; Logistic Regression was
auto-selected for the best F1 score.

## 3. Seed the database

```bash
python3 seed_db.py                # imports ALL ~100k cleaned students
python3 seed_db.py --limit 3000   # or a smaller demo subset
```

This creates `instance/smartedu.db` (SQLite) with two demo accounts:

- **Admin:** admin@smartedu.com / admin123
- **Teacher:** teacher@smartedu.com / teacher123

## 4. Run the app

```bash
python3 app.py
```

Visit **http://localhost:5000** and log in with a demo account.

## Notes on the data cleaning

The raw CSV had: inconsistent casing (`STU12345` vs `stu12345`), stray
whitespace, mixed Result labels (`Pass`/`PASS`/`pass`/`P`), out-of-range
values (attendance of 150% or 999%, negative marks), missing fields, and
1,981 duplicate roll numbers. `ml/clean_data.py` standardizes all of this,
imputes missing numeric values with column medians, and derives a
`Risk_Level` label (used to train the ML model) from a weighted score across
attendance, quiz, assignment, midterm and final marks.

## Architecture

- **Backend:** Flask, Flask-Login, Flask-SQLAlchemy, SQLite
- **ML:** scikit-learn (Logistic Regression, Random Forest), XGBoost
- **Frontend:** Bootstrap 5, Chart.js, vanilla JS
- **Reports:** ReportLab (PDF), built-in csv module (CSV)


