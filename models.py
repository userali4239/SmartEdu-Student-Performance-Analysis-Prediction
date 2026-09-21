from datetime import datetime, date
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="teacher")  # teacher | admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == "admin"


class Department(db.Model):
    __tablename__ = "departments"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)


class Student(db.Model):
    __tablename__ = "students"
    id = db.Column(db.Integer, primary_key=True)
    roll_number = db.Column(db.String(20), unique=True, nullable=False, index=True)
    name = db.Column(db.String(150), nullable=False)
    father_name = db.Column(db.String(150))
    gender = db.Column(db.String(10))
    dob = db.Column(db.Date)
    department = db.Column(db.String(100), default="Computer Science")
    semester = db.Column(db.Integer, default=1)
    section = db.Column(db.String(10), default="A")
    session = db.Column(db.String(20), default="2023-2027")
    phone = db.Column(db.String(20))
    email = db.Column(db.String(120))

    # Academic data (matches ML feature set)
    attendance_percentage = db.Column(db.Float, default=0)
    quiz_marks = db.Column(db.Float, default=0)          # out of 20
    assignment_marks = db.Column(db.Float, default=0)    # out of 20
    midterm_marks = db.Column(db.Float, default=0)       # out of 30
    final_marks = db.Column(db.Float, default=0)         # out of 30
    result = db.Column(db.String(10))                    # Pass / Fail

    # ML prediction (cached)
    risk_level = db.Column(db.String(20))                # Low / Medium / High Risk
    confidence = db.Column(db.Float)
    predicted_at = db.Column(db.DateTime)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def total_marks(self):
        return round((self.quiz_marks or 0) + (self.assignment_marks or 0) +
                     (self.midterm_marks or 0) + (self.final_marks or 0), 1)

    @property
    def risk_badge_class(self):
        mapping = {"Low Risk": "success", "Medium Risk": "warning", "High Risk": "danger"}
        return mapping.get(self.risk_level, "secondary")


class AttendanceLog(db.Model):
    __tablename__ = "attendance_logs"
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    present = db.Column(db.Boolean, nullable=False, default=True)

    student = db.relationship("Student", backref=db.backref("attendance_logs", lazy="dynamic"))


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    risk_level = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    model_used = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    student = db.relationship("Student", backref=db.backref("predictions", lazy="dynamic"))
