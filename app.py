import os
import io
import csv
from datetime import datetime, date

from flask import (Flask, render_template, redirect, url_for, request,
                    flash, session, send_file, jsonify, abort)
from flask_login import (login_user, logout_user, login_required,
                          current_user, LoginManager)
from sqlalchemy import func, or_

from config import Config
from extensions import db, login_manager
from models import User, Student, AttendanceLog, Prediction, Department
import ml_utils


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    register_routes(app)
    return app


def register_routes(app):

    # ---------------------------------------------------------------- AUTH
    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            remember = bool(request.form.get("remember"))
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                login_user(user, remember=remember)
                flash(f"Welcome back, {user.name}!", "success")
                next_page = request.args.get("next")
                return redirect(next_page or url_for("dashboard"))
            flash("Invalid email or password.", "danger")
        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        flash("You have been logged out.", "info")
        return redirect(url_for("login"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            role = request.form.get("role", "teacher")
            if User.query.filter_by(email=email).first():
                flash("An account with this email already exists.", "danger")
                return redirect(url_for("register"))
            user = User(name=name, email=email, role=role)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for("login"))
        return render_template("register.html")

    # ----------------------------------------------------------- DASHBOARD
    @app.route("/dashboard")
    @login_required
    def dashboard():
        total_students = Student.query.count()
        avg_attendance = db.session.query(func.avg(Student.attendance_percentage)).scalar() or 0

        # Average of the four components as "average marks %"
        avg_quiz = db.session.query(func.avg(Student.quiz_marks)).scalar() or 0
        avg_assign = db.session.query(func.avg(Student.assignment_marks)).scalar() or 0
        avg_mid = db.session.query(func.avg(Student.midterm_marks)).scalar() or 0
        avg_final = db.session.query(func.avg(Student.final_marks)).scalar() or 0
        avg_marks_pct = round(((avg_quiz / 20) + (avg_assign / 20) + (avg_mid / 30) + (avg_final / 30)) / 4 * 100, 1)

        risk_counts = dict(
            db.session.query(Student.risk_level, func.count(Student.id))
            .group_by(Student.risk_level).all()
        )
        high_risk = risk_counts.get("High Risk", 0)
        medium_risk = risk_counts.get("Medium Risk", 0)
        low_risk = risk_counts.get("Low Risk", 0)

        total_classes = db.session.query(func.count(func.distinct(Student.section))).scalar() or 0

        # Department distribution for a bar chart
        dept_counts = dict(
            db.session.query(Student.department, func.count(Student.id))
            .group_by(Student.department).all()
        )

        # Recent high risk students for notification panel
        high_risk_students = (Student.query.filter_by(risk_level="High Risk")
                               .order_by(Student.predicted_at.desc().nullslast())
                               .limit(8).all())
        low_attendance_students = (Student.query.filter(Student.attendance_percentage < 75)
                                    .order_by(Student.attendance_percentage.asc())
                                    .limit(8).all())

        return render_template(
            "dashboard.html",
            total_students=total_students,
            avg_attendance=round(avg_attendance, 1),
            avg_marks_pct=avg_marks_pct,
            high_risk=high_risk, medium_risk=medium_risk, low_risk=low_risk,
            total_classes=total_classes,
            dept_counts=dept_counts,
            high_risk_students=high_risk_students,
            low_attendance_students=low_attendance_students,
            ml_metrics=ml_utils.get_metrics(),
        )

    # --------------------------------------------------------- STUDENTS
    @app.route("/students")
    @login_required
    def students_list():
        q = request.args.get("q", "").strip()
        risk_filter = request.args.get("risk", "")
        dept_filter = request.args.get("department", "")
        page = request.args.get("page", 1, type=int)

        query = Student.query
        if q:
            like = f"%{q}%"
            query = query.filter(or_(Student.name.ilike(like), Student.roll_number.ilike(like)))
        if risk_filter:
            query = query.filter(Student.risk_level == risk_filter)
        if dept_filter:
            query = query.filter(Student.department == dept_filter)

        query = query.order_by(Student.roll_number.asc())
        pagination = query.paginate(page=page, per_page=app.config["STUDENTS_PER_PAGE"], error_out=False)

        departments = [d[0] for d in db.session.query(Student.department).distinct().all() if d[0]]

        return render_template("students.html", pagination=pagination, students=pagination.items,
                                q=q, risk_filter=risk_filter, dept_filter=dept_filter,
                                departments=departments)

    @app.route("/students/add", methods=["GET", "POST"])
    @login_required
    def student_add():
        if request.method == "POST":
            roll_number = request.form.get("roll_number", "").strip().upper()
            if Student.query.filter_by(roll_number=roll_number).first():
                flash("A student with this Roll Number already exists.", "danger")
                return redirect(url_for("student_add"))

            student = Student(
                roll_number=roll_number,
                name=request.form.get("name", "").strip(),
                father_name=request.form.get("father_name", "").strip(),
                gender=request.form.get("gender"),
                department=request.form.get("department", "Computer Science"),
                semester=request.form.get("semester", 1, type=int),
                section=request.form.get("section", "A"),
                session=request.form.get("session", ""),
                phone=request.form.get("phone", ""),
                email=request.form.get("email", ""),
                attendance_percentage=request.form.get("attendance_percentage", 0, type=float),
                quiz_marks=request.form.get("quiz_marks", 0, type=float),
                assignment_marks=request.form.get("assignment_marks", 0, type=float),
                midterm_marks=request.form.get("midterm_marks", 0, type=float),
                final_marks=request.form.get("final_marks", 0, type=float),
            )
            student.result = "Pass" if student.final_marks >= 15 else "Fail"
            db.session.add(student)
            db.session.commit()
            _run_prediction(student)
            flash(f"Student {student.name} added successfully.", "success")
            return redirect(url_for("students_list"))
        return render_template("student_form.html", student=None)

    @app.route("/students/<int:student_id>")
    @login_required
    def student_detail(student_id):
        student = db.session.get(Student, student_id) or abort(404)
        recommendations = ml_utils.get_recommendations(student.risk_level) if student.risk_level else []
        recent_predictions = student.predictions.order_by(Prediction.created_at.desc()).limit(5).all()
        attendance_logs = student.attendance_logs.order_by(AttendanceLog.date.desc()).limit(10).all()
        return render_template("student_detail.html", student=student,
                                recommendations=recommendations,
                                recent_predictions=recent_predictions,
                                attendance_logs=attendance_logs)

    @app.route("/students/<int:student_id>/edit", methods=["GET", "POST"])
    @login_required
    def student_edit(student_id):
        student = db.session.get(Student, student_id) or abort(404)
        if request.method == "POST":
            student.name = request.form.get("name", "").strip()
            student.father_name = request.form.get("father_name", "").strip()
            student.gender = request.form.get("gender")
            student.department = request.form.get("department", student.department)
            student.semester = request.form.get("semester", student.semester, type=int)
            student.section = request.form.get("section", student.section)
            student.session = request.form.get("session", student.session)
            student.phone = request.form.get("phone", "")
            student.email = request.form.get("email", "")
            student.attendance_percentage = request.form.get("attendance_percentage", 0, type=float)
            student.quiz_marks = request.form.get("quiz_marks", 0, type=float)
            student.assignment_marks = request.form.get("assignment_marks", 0, type=float)
            student.midterm_marks = request.form.get("midterm_marks", 0, type=float)
            student.final_marks = request.form.get("final_marks", 0, type=float)
            student.result = "Pass" if student.final_marks >= 15 else "Fail"
            db.session.commit()
            _run_prediction(student)
            flash("Student updated successfully.", "success")
            return redirect(url_for("student_detail", student_id=student.id))
        return render_template("student_form.html", student=student)

    @app.route("/students/<int:student_id>/delete", methods=["POST"])
    @login_required
    def student_delete(student_id):
        student = db.session.get(Student, student_id) or abort(404)
        db.session.delete(student)
        db.session.commit()
        flash("Student deleted.", "info")
        return redirect(url_for("students_list"))

    @app.route("/students/<int:student_id>/predict", methods=["POST"])
    @login_required
    def student_predict(student_id):
        student = db.session.get(Student, student_id) or abort(404)
        _run_prediction(student)
        flash(f"Prediction updated: {student.risk_level} ({student.confidence}% confidence)", "success")
        return redirect(url_for("student_detail", student_id=student.id))

    # -------------------------------------------------------- ATTENDANCE
    @app.route("/attendance", methods=["GET", "POST"])
    @login_required
    def attendance():
        if request.method == "POST":
            student_id = request.form.get("student_id", type=int)
            student = db.session.get(Student, student_id)
            if not student:
                flash("Student not found.", "danger")
                return redirect(url_for("attendance"))
            log = AttendanceLog(
                student_id=student.id,
                subject=request.form.get("subject", "General"),
                date=datetime.strptime(request.form.get("date"), "%Y-%m-%d").date(),
                present=bool(request.form.get("present")),
            )
            db.session.add(log)
            db.session.commit()

            # Recalculate attendance percentage from logs
            logs = student.attendance_logs.all()
            if logs:
                pct = sum(1 for l in logs if l.present) / len(logs) * 100
                student.attendance_percentage = round(pct, 1)
                db.session.commit()

            flash("Attendance recorded.", "success")
            return redirect(url_for("student_detail", student_id=student.id))

        q = request.args.get("q", "").strip()
        results = []
        if q:
            like = f"%{q}%"
            results = Student.query.filter(or_(Student.name.ilike(like), Student.roll_number.ilike(like))).limit(10).all()
        today = date.today().isoformat()
        return render_template("attendance.html", results=results, q=q, today=today)

    # ------------------------------------------------------------ REPORTS
    @app.route("/reports")
    @login_required
    def reports():
        return render_template("reports.html")

    @app.route("/reports/export/csv")
    @login_required
    def export_csv():
        risk_filter = request.args.get("risk", "")
        query = Student.query
        if risk_filter:
            query = query.filter(Student.risk_level == risk_filter)
        students = query.order_by(Student.roll_number).all()

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Roll Number", "Name", "Department", "Semester", "Attendance %",
                          "Quiz", "Assignment", "Midterm", "Final", "Result", "Risk Level", "Confidence"])
        for s in students:
            writer.writerow([s.roll_number, s.name, s.department, s.semester, s.attendance_percentage,
                              s.quiz_marks, s.assignment_marks, s.midterm_marks, s.final_marks,
                              s.result, s.risk_level, s.confidence])
        mem = io.BytesIO(buf.getvalue().encode("utf-8"))
        filename = f"smartedu_report_{risk_filter or 'all'}_{date.today().isoformat()}.csv"
        return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=filename)

    @app.route("/reports/export/pdf/<int:student_id>")
    @login_required
    def export_pdf(student_id):
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        student = db.session.get(Student, student_id) or abort(404)
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [Paragraph("SmartEdu - Individual Student Report", styles["Title"]), Spacer(1, 12)]

        info_data = [
            ["Roll Number", student.roll_number, "Name", student.name],
            ["Department", student.department, "Semester", str(student.semester)],
            ["Section", student.section, "Session", student.session or "-"],
        ]
        info_table = Table(info_data, colWidths=[3 * cm, 4 * cm, 3 * cm, 4 * cm])
        info_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                         ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                                         ("BACKGROUND", (2, 0), (2, -1), colors.whitesmoke)]))
        elements += [info_table, Spacer(1, 16)]

        marks_data = [
            ["Attendance %", "Quiz", "Assignment", "Midterm", "Final", "Result"],
            [student.attendance_percentage, student.quiz_marks, student.assignment_marks,
             student.midterm_marks, student.final_marks, student.result],
        ]
        marks_table = Table(marks_data, colWidths=[3 * cm] * 6)
        marks_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
                                          ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]))
        elements += [marks_table, Spacer(1, 16)]

        elements.append(Paragraph(f"<b>Predicted Risk Level:</b> {student.risk_level or 'Not yet predicted'} "
                                   f"({student.confidence or 0}% confidence)", styles["Normal"]))
        elements.append(Spacer(1, 8))

        recs = ml_utils.get_recommendations(student.risk_level) if student.risk_level else []
        if recs:
            elements.append(Paragraph("<b>Recommendations:</b>", styles["Normal"]))
            for r in recs:
                elements.append(Paragraph(f"- {r}", styles["Normal"]))

        doc.build(elements)
        buf.seek(0)
        return send_file(buf, mimetype="application/pdf", as_attachment=True,
                          download_name=f"{student.roll_number}_report.pdf")

    # ------------------------------------------------------------ ADMIN
    @app.route("/admin/users")
    @login_required
    def admin_users():
        if not current_user.is_admin:
            abort(403)
        users = User.query.order_by(User.created_at.desc()).all()
        return render_template("admin_users.html", users=users)

    @app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
    @login_required
    def admin_user_delete(user_id):
        if not current_user.is_admin:
            abort(403)
        user = db.session.get(User, user_id) or abort(404)
        if user.id == current_user.id:
            flash("You cannot delete your own account.", "warning")
        else:
            db.session.delete(user)
            db.session.commit()
            flash("User removed.", "info")
        return redirect(url_for("admin_users"))

    # ------------------------------------------------------------- API
    @app.route("/api/dashboard-charts")
    @login_required
    def api_dashboard_charts():
        risk_counts = dict(
            db.session.query(Student.risk_level, func.count(Student.id))
            .group_by(Student.risk_level).all()
        )
        attendance_buckets = {"0-50": 0, "50-75": 0, "75-90": 0, "90-100": 0}
        for (pct,) in db.session.query(Student.attendance_percentage).all():
            if pct < 50:
                attendance_buckets["0-50"] += 1
            elif pct < 75:
                attendance_buckets["50-75"] += 1
            elif pct < 90:
                attendance_buckets["75-90"] += 1
            else:
                attendance_buckets["90-100"] += 1

        marks_avg = {
            "Quiz": round((db.session.query(func.avg(Student.quiz_marks)).scalar() or 0), 1),
            "Assignment": round((db.session.query(func.avg(Student.assignment_marks)).scalar() or 0), 1),
            "Midterm": round((db.session.query(func.avg(Student.midterm_marks)).scalar() or 0), 1),
            "Final": round((db.session.query(func.avg(Student.final_marks)).scalar() or 0), 1),
        }

        return jsonify({
            "risk": risk_counts,
            "attendance_buckets": attendance_buckets,
            "marks_avg": marks_avg,
        })


def _run_prediction(student: Student):
    risk_level, confidence = ml_utils.predict_risk(
        student.attendance_percentage, student.quiz_marks,
        student.assignment_marks, student.midterm_marks
    )
    student.risk_level = risk_level
    student.confidence = confidence
    student.predicted_at = datetime.utcnow()
    db.session.add(Prediction(student_id=student.id, risk_level=risk_level,
                               confidence=confidence, model_used=ml_utils.get_metrics()["best_model"]))
    db.session.commit()


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
