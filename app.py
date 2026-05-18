import json
import os
import re
import secrets
import sqlite3
import struct
import subprocess
import sys
import uuid
from datetime import datetime
from io import BytesIO
from functools import wraps

import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from flask_login import LoginManager, current_user, login_required
from werkzeug.security import generate_password_hash

from auth.auth_routes import auth_bp
from auth.user_model import User
from chatbot.chatbot_logic import get_smart_bot_response
from config import BASE_DIR, CHAT_MEMORY_LIMIT, SECRET_KEY, DATABASE as CONFIG_DATABASE, DATABASE_URL as CONFIG_DATABASE_URL
from grievance.grievance_routes import grievance_bp
from src.db import connect, execute, fetchall, fetchone, fetch_value, get_backend, identity_column, last_insert_id
from src.feature_catalog import enrich_feature_schema, get_feature_label
from src.guidance import get_guidance
from src.predict import predict_with_explain
from src.email_utils import send_email
from src.report_reader import format_missing_fields, parse_uploaded_report
from src.report import generate_report
from src.train import MODEL_CONFIG
from migrations import run_migrations

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
DATABASE = CONFIG_DATABASE
DATABASE_URL = CONFIG_DATABASE_URL

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth.login"

app.register_blueprint(auth_bp)
app.register_blueprint(grievance_bp)


def csrf_token():
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


app.jinja_env.globals["csrf_token"] = csrf_token
app.jinja_env.globals["now"] = datetime.now


@app.before_request
def protect_from_csrf():
    if app.config.get("TESTING") or request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return None
    if request.path.startswith("/api/predict/"):
        return None
    token = request.form.get("csrf_token") or request.headers.get("X-CSRFToken")
    if not token or token != session.get("_csrf_token"):
        return jsonify({"error": "CSRF token missing or invalid"}), 400
    return None

def get_db():
    if DATABASE != CONFIG_DATABASE:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        return conn
    return connect()


def to_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, bytes):
        try:
            if len(value) == 4:
                return float(struct.unpack("<f", value)[0])
            if len(value) == 8:
                return float(struct.unpack("<d", value)[0])
        except struct.error:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_prediction_row(row):
    data = dict(row)
    data["probability"] = to_float(data.get("probability"))
    data["confidence"] = to_float(
        data.get("confidence"),
        max(data["probability"], 1 - data["probability"]),
    )
    return data


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if getattr(current_user, "role", None) != "admin":
            if request.path.startswith("/api/"):
                return jsonify({"error": "Admin access required"}), 403
            flash("Admin access required. Your account is not an admin account.")
            return redirect(url_for("patient_dashboard"))
        return view(*args, **kwargs)

    return wrapped


@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    user = fetchone(execute(conn, "SELECT * FROM users WHERE id=?", (user_id,)))
    conn.close()

    if user:
        return User(user["id"], user["name"], user["email"], user["role"])
    return None


def init_db():
    conn = get_db()
    execute(
        conn,
        f"""
        CREATE TABLE IF NOT EXISTS users(
            id {identity_column()},
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'patient'
        )
        """,
    )

    execute(
        conn,
        f"""
        CREATE TABLE IF NOT EXISTS grievances(
            id {identity_column()},
            user_id INTEGER,
            message TEXT NOT NULL,
            status TEXT DEFAULT 'Pending',
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    )

    execute(conn, "UPDATE users SET email=lower(email)")
    admin_count = fetch_value(conn, "SELECT COUNT(*) FROM users WHERE role='admin'") or 0
    user_count = fetch_value(conn, "SELECT COUNT(*) FROM users") or 0
    if user_count and not admin_count:
        execute(
            conn,
            "UPDATE users SET role='admin' WHERE id=(SELECT MIN(id) FROM users)"
        )

    conn.commit()
    conn.close()
    run_migrations(DATABASE)


def parse_features(payload, disease):
    if not payload or "features" not in payload:
        raise ValueError("Request JSON must include a features array.")

    features = payload["features"]
    if not isinstance(features, list):
        raise ValueError("features must be a list.")

    feature_schema = load_feature_schema(disease)
    expected_count = len(feature_schema)
    if len(features) != expected_count:
        raise ValueError(
            f"{disease.title()} prediction requires {expected_count} features."
        )

    parsed = []
    for spec, value in zip(feature_schema, features):
        if spec.get("kind") == "number":
            parsed.append(float(value))
        else:
            parsed.append("" if value is None else str(value).strip())
    return parsed


def features_to_dict(disease, features):
    return {
        get_feature_label(disease, spec["name"]): value
        for spec, value in zip(load_feature_schema(disease), features)
    }


def load_feature_schema(disease):
    import joblib

    try:
        raw_schema = joblib.load(MODEL_CONFIG[disease]["features_path"])
    except FileNotFoundError:
        raw_schema = [
            {"name": name, "kind": "number", "label": name}
            for name in MODEL_CONFIG[disease].get("features", [])
        ]
    return enrich_feature_schema(disease, raw_schema)


def load_feature_names(disease):
    return [item["name"] for item in load_feature_schema(disease)]


def disease_label(disease):
    return disease.replace("_", " ").title()


def load_model_metrics(disease):
    try:
        with open(MODEL_CONFIG[disease]["metrics_path"], encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def build_disease_forms():
    forms = []
    for disease in MODEL_CONFIG:
        forms.append({
            "key": disease,
            "label": disease_label(disease),
            "fields": load_feature_schema(disease),
        })
    return forms


def save_prediction(user_id, disease, probability, risk, confidence, features, batch_id=None):
    conn = get_db()
    insert_sql = """
    INSERT INTO predictions (user_id, disease, probability, risk, confidence, feature_values, model_version, batch_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    if get_backend() == "postgresql":
        insert_sql += " RETURNING id"
    cursor = execute(
        conn,
        insert_sql,
        (
            user_id,
            disease,
            probability,
            risk,
            confidence,
            json.dumps(features_to_dict(disease, features)),
            MODEL_CONFIG[disease]["version"],
            batch_id,
        ),
    )
    conn.commit()
    prediction_id = last_insert_id(cursor, "predictions")
    conn.close()
    return prediction_id


def prediction_response(disease):
    try:
        features = parse_features(request.get_json(silent=True), disease)
        probability, risk, confidence, explanation = predict_with_explain(features, disease)
        metrics = load_model_metrics(disease)
        guidance = get_guidance(disease, risk)
        prediction_id = save_prediction(
            current_user.id if current_user.is_authenticated else None,
            disease,
            probability,
            risk,
            confidence,
            features,
        )

        return jsonify({
            "id": prediction_id,
            "disease": disease,
            "probability": probability,
            "risk": risk,
            "confidence": confidence,
            "explanation": explanation,
            "model_metrics": metrics,
            "guidance": guidance,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"error": "Model files are missing. Run python src/train.py."}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def read_uploaded_features(file_storage, disease):
    if not file_storage or not file_storage.filename:
        raise ValueError("Upload a CSV, Excel, PDF, TXT, PNG, or JPG file.")

    content = file_storage.read()
    filename = file_storage.filename.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(BytesIO(content))
    elif filename.endswith((".pdf", ".png", ".jpg", ".jpeg", ".txt")):
        return read_pdf_features(content, disease, filename)
    else:
        raise ValueError("Only CSV, Excel, PDF, TXT, PNG, and JPG files are supported.")

    if df.empty:
        raise ValueError("Uploaded file has no rows.")

    feature_names = load_feature_names(disease)
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    schema = load_feature_schema(disease)
    return [
        float(df.iloc[0][spec["name"]]) if spec["kind"] == "number" else str(df.iloc[0][spec["name"]]).strip()
        for spec in schema
    ]


def read_uploaded_feature_rows(file_storage, disease):
    if not file_storage or not file_storage.filename:
        raise ValueError("Upload a CSV, Excel, PDF, TXT, PNG, or JPG file.")

    content = file_storage.read()
    filename = file_storage.filename.lower()
    if filename.endswith((".pdf", ".png", ".jpg", ".jpeg", ".txt")):
        return [read_pdf_features(content, disease, filename)]
    if filename.endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(BytesIO(content))
    else:
        raise ValueError("Only CSV, Excel, PDF, TXT, PNG, and JPG files are supported.")

    if df.empty:
        raise ValueError("Uploaded file has no rows.")

    feature_names = load_feature_names(disease)
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    schema = load_feature_schema(disease)
    rows = []
    for _, row in df.iterrows():
        rows.append([
            float(row[spec["name"]]) if spec["kind"] == "number" else str(row[spec["name"]]).strip()
            for spec in schema
        ])
    return rows


def read_pdf_features(content, disease, filename="report.pdf"):
    feature_schema = load_feature_schema(disease)
    values = parse_uploaded_report(content, filename, disease, feature_schema)
    missing = format_missing_fields(disease, feature_schema, values)
    if missing:
        raise ValueError(
            "Could not extract all required report values. Missing: "
            + ", ".join(missing)
        )

    return [values[item["name"]] for item in feature_schema]


def get_recent_chat_history(user_id, limit=CHAT_MEMORY_LIMIT):
    conn = get_db()
    history = fetchall(
        execute(
            conn,
            """
            SELECT role, message, source, created_at
            FROM chat_messages
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
    )
    conn.close()
    return list(reversed(history))


def save_chat_message(user_id, role, message, source="ui"):
    conn = get_db()
    execute(
        conn,
        """
        INSERT INTO chat_messages (user_id, role, message, source)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, role, message, source),
    )
    conn.commit()
    conn.close()


def clear_chat_history(user_id):
    conn = get_db()
    execute(conn, "DELETE FROM chat_messages WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def _appointment_datetime(appointment_date, time_slot):
    try:
        return datetime.strptime(f"{appointment_date} {time_slot}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def purge_expired_appointments(user_id=None):
    conn = get_db()
    params = ()
    sql = """
        SELECT id, appointment_date, time_slot
        FROM appointments
    """
    if user_id is not None:
        sql += " WHERE user_id=?"
        params = (user_id,)
    rows = fetchall(execute(conn, sql, params))
    now = datetime.now()
    expired_ids = []
    for row in rows:
        appointment_at = _appointment_datetime(row["appointment_date"], row["time_slot"])
        if appointment_at and appointment_at < now:
            expired_ids.append(row["id"])

    if expired_ids:
        placeholders = ",".join("?" for _ in expired_ids)
        execute(conn, f"DELETE FROM appointments WHERE id IN ({placeholders})", tuple(expired_ids))
        conn.commit()
    conn.close()
    return len(expired_ids)


def get_patient_analytics(user_id):
    purge_expired_appointments(user_id)
    conn = get_db()
    predictions = fetchall(
        execute(
            conn,
            """
            SELECT disease, risk, probability, confidence, created_at
            FROM predictions
            WHERE user_id=?
            ORDER BY created_at ASC
            """,
            (user_id,),
        )
    )
    appointments = fetchall(
        execute(
            conn,
            """
            SELECT id, doctor_name, specialty, appointment_date, time_slot, mode, status, notes
            FROM appointments
            WHERE user_id=?
            ORDER BY appointment_date ASC, time_slot ASC
            LIMIT 6
            """,
            (user_id,),
        )
    )
    conn.close()

    disease_counts = {}
    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    trend = []
    for item in predictions:
        disease_counts[item["disease"].title()] = disease_counts.get(item["disease"].title(), 0) + 1
        risk_counts[item["risk"]] = risk_counts.get(item["risk"], 0) + 1
        trend.append({
            "label": item["created_at"],
            "probability": round(to_float(item.get("probability")) * 100, 2),
        })

    return {
        "total_predictions": len(predictions),
        "high_risk_cases": risk_counts.get("High", 0),
        "active_appointments": len([item for item in appointments if item["status"] == "Booked"]),
        "disease_counts": disease_counts,
        "risk_counts": risk_counts,
        "trend": trend[-8:],
        "appointments": appointments,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
@login_required
def dashboard():
    if current_user.role == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("patient_dashboard"))


@app.route("/patient/dashboard")
@login_required
def patient_dashboard():
    purge_expired_appointments(current_user.id)
    conn = get_db()
    history = fetchall(
        execute(
            conn,
            """
            SELECT id, disease, probability, risk, confidence, created_at
            FROM predictions
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 8
            """,
            (current_user.id,),
        )
    )
    conn.close()
    history = [normalize_prediction_row(row) for row in history]
    analytics = get_patient_analytics(current_user.id)
    return render_template(
        "patient_dashboard.html",
        disease_forms=build_disease_forms(),
        history=history,
        analytics=analytics,
    )


@app.route("/account/password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        conn = get_db()
        user = fetchone(execute(conn, "SELECT * FROM users WHERE id=?", (current_user.id,)))
        if not user:
            conn.close()
            flash("Account not found.")
            return redirect(url_for("change_password"))
        from werkzeug.security import check_password_hash
        if not check_password_hash(user["password"], current_password):
            conn.close()
            flash("Current password is incorrect.")
            return redirect(url_for("change_password"))
        execute(
            conn,
            "UPDATE users SET password=? WHERE id=?",
            (generate_password_hash(new_password), current_user.id),
        )
        conn.commit()
        conn.close()
        flash("Password changed.")
        return redirect(url_for("dashboard"))
    return render_template("change_password.html")


@app.route("/chatbot")
@login_required
def chatbot_page():
    return render_template("chatbot.html", chat_history=get_recent_chat_history(current_user.id))


@app.route("/result")
@login_required
def result():
    return render_template("result.html")


@app.route("/batch-result")
@login_required
def batch_result():
    return render_template("batch_result.html")


@app.route("/admin")
@login_required
def admin_shortcut():
    if current_user.role != "admin":
        flash("Admin access required. Your account is not an admin account.")
        return redirect(url_for("patient_dashboard"))
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    conn = get_db()
    total_users = fetch_value(conn, "SELECT COUNT(*) FROM users") or 0
    total_predictions = fetch_value(conn, "SELECT COUNT(*) FROM predictions") or 0
    total_grievances = fetch_value(conn, "SELECT COUNT(*) FROM grievances") or 0

    prediction_rows = fetchall(
        execute(
            conn,
            """
            SELECT p.id, u.email, p.disease, p.probability, p.risk, p.confidence,
                   p.clinical_status, p.final_diagnosis, p.model_version, p.created_at
            FROM predictions p
            LEFT JOIN users u ON u.id = p.user_id
            ORDER BY p.created_at DESC
            LIMIT 10
            """
        )
    )
    conn.close()
    prediction_rows = [normalize_prediction_row(row) for row in prediction_rows]

    metrics = {}
    for disease, config in MODEL_CONFIG.items():
        metrics[disease] = load_model_metrics(disease)

    return render_template(
        "admin.html",
        total_users=total_users,
        total_predictions=total_predictions,
        total_grievances=total_grievances,
        predictions=prediction_rows,
        metrics=metrics,
    )


@app.route("/admin/grievances")
@admin_required
def admin_grievances():
    conn = get_db()
    grievances = fetchall(
        execute(
            conn,
            """
            SELECT g.id, u.email, g.message, g.status, g.response, g.created_at
            FROM grievances g
            LEFT JOIN users u ON u.id = g.user_id
            ORDER BY g.created_at DESC
            """
        )
    )
    conn.close()
    return render_template("admin_grievance.html", grievances=grievances)


@app.route("/admin/grievances/<int:grievance_id>", methods=["POST"])
@admin_required
def update_grievance(grievance_id):
    status = request.form.get("status", "Pending")
    if status not in {"Pending", "In Progress", "Resolved"}:
        flash("Invalid grievance status.")
        return redirect(url_for("admin_grievances"))
    response = request.form.get("response", "")
    conn = get_db()
    execute(
        conn,
        "UPDATE grievances SET status=?, response=? WHERE id=?",
        (status, response, grievance_id),
    )
    conn.commit()
    conn.close()
    flash("Grievance updated.")
    return redirect(url_for("admin_grievances"))


@app.route("/admin/users", methods=["GET", "POST"])
@admin_required
def admin_users():
    conn = get_db()
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "patient")
        if role not in {"patient", "admin"}:
            flash("Invalid role.")
        elif not name or not email or not password:
            flash("Name, email, and password are required.")
        else:
            try:
                execute(
                    conn,
                    "INSERT INTO users (name,email,password,role) VALUES (?,?,?,?)",
                    (name, email, generate_password_hash(password), role),
                )
                conn.commit()
                flash("User created.")
            except Exception as exc:
                if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                    flash("Email already exists.")
                else:
                    raise
    users = fetchall(execute(conn, "SELECT id, name, email, role FROM users ORDER BY id"))
    conn.close()
    return render_template("admin_users.html", users=users)


@app.route("/admin/users/<int:user_id>/role", methods=["POST"])
@admin_required
def admin_update_user_role(user_id):
    role = request.form.get("role", "patient")
    if role not in {"patient", "admin"}:
        flash("Invalid role.")
        return redirect(url_for("admin_users"))
    conn = get_db()
    execute(conn, "UPDATE users SET role=? WHERE id=?", (role, user_id))
    conn.commit()
    conn.close()
    flash("User role updated.")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<int:user_id>/reset-password", methods=["POST"])
@admin_required
def admin_reset_user_password(user_id):
    new_password = request.form.get("new_password", "")
    if len(new_password) < 6:
        flash("Password must be at least 6 characters.")
        return redirect(url_for("admin_users"))
    conn = get_db()
    execute(
        conn,
        "UPDATE users SET password=? WHERE id=?",
        (generate_password_hash(new_password), user_id),
    )
    conn.commit()
    conn.close()
    flash("Password reset.")
    return redirect(url_for("admin_users"))


@app.route("/admin/predictions/<int:prediction_id>/validate", methods=["POST"])
@admin_required
def validate_prediction(prediction_id):
    status = request.form.get("clinical_status", "Pending Review")
    if status not in {"Pending Review", "Reviewed", "Confirmed", "Rejected"}:
        flash("Invalid clinical status.")
        return redirect(url_for("admin_dashboard"))
    conn = get_db()
    execute(
        conn,
        """
        UPDATE predictions
        SET clinical_status=?, final_diagnosis=?, clinical_notes=?,
            actual_result=?, validated_by=?, validated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (
            status,
            request.form.get("final_diagnosis", ""),
            request.form.get("clinical_notes", ""),
            request.form.get("actual_result") or None,
            current_user.id,
            prediction_id,
        ),
    )
    conn.commit()
    conn.close()
    flash("Prediction review saved.")
    return redirect(url_for("admin_dashboard"))


@app.route("/appointments/book", methods=["POST"])
@login_required
def book_appointment():
    doctor_name = request.form.get("doctor_name", "").strip()
    specialty = request.form.get("specialty", "").strip()
    appointment_date = request.form.get("appointment_date", "").strip()
    time_slot = request.form.get("time_slot", "").strip()
    mode = request.form.get("mode", "Video").strip() or "Video"
    notes = request.form.get("notes", "").strip()

    if not all([doctor_name, specialty, appointment_date, time_slot]):
        flash("Doctor, specialty, date, and time slot are required.")
        return redirect(url_for("patient_dashboard"))

    appointment_at = _appointment_datetime(appointment_date, time_slot)
    if not appointment_at:
        flash("Enter a valid appointment date and time.")
        return redirect(url_for("patient_dashboard"))
    if appointment_at <= datetime.now():
        flash("Choose a future appointment slot.")
        return redirect(url_for("patient_dashboard"))

    conn = get_db()
    execute(
        conn,
        """
        INSERT INTO appointments (user_id, doctor_name, specialty, appointment_date, time_slot, mode, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (current_user.id, doctor_name, specialty, appointment_date, time_slot, mode, notes),
    )
    conn.commit()
    conn.close()
    flash("Appointment booked successfully.")
    return redirect(url_for("patient_dashboard"))


@app.route("/api/analytics/patient")
@login_required
def patient_analytics_api():
    return jsonify(get_patient_analytics(current_user.id))


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Message is required."}), 400
    history = get_recent_chat_history(current_user.id)
    result = get_smart_bot_response(message, history=history, state=session.get("chat_state"))
    session["chat_state"] = result["state"]
    save_chat_message(current_user.id, "user", message, source="ui")
    save_chat_message(current_user.id, "assistant", result["reply"], source=result.get("source", "ui"))
    return jsonify({"reply": result["reply"], "done": result["done"], "source": result.get("source")})


@app.route("/api/chat/clear", methods=["POST"])
@login_required
def clear_chat():
    clear_chat_history(current_user.id)
    session.pop("chat_state", None)
    return jsonify({"message": "Chat cleared."})


@app.route("/api/predict/diabetes", methods=["POST"])
@login_required
def predict_diabetes():
    return prediction_response("diabetes")


@app.route("/api/predict/heart", methods=["POST"])
@login_required
def predict_heart():
    return prediction_response("heart")


@app.route("/api/predict/<disease>", methods=["POST"])
@login_required
def predict_generic(disease):
    if disease not in MODEL_CONFIG:
        return jsonify({"error": "Unsupported disease."}), 404
    return prediction_response(disease)


@app.route("/api/predict/<disease>/upload", methods=["POST"])
@login_required
def predict_upload(disease):
    if disease not in MODEL_CONFIG:
        return jsonify({"error": "Unsupported disease."}), 404
    try:
        rows = read_uploaded_feature_rows(request.files.get("file"), disease)
        batch_id = str(uuid.uuid4())
        results = []
        for features in rows:
            probability, risk, confidence, explanation = predict_with_explain(features, disease)
            prediction_id = save_prediction(
                current_user.id, disease, probability, risk, confidence, features, batch_id
            )
            results.append({
                "id": prediction_id,
                "disease": disease,
                "probability": probability,
                "risk": risk,
                "confidence": confidence,
                "explanation": explanation,
            })
        if len(results) == 1:
            return jsonify(results[0])
        return jsonify({"batch": True, "batch_id": batch_id, "results": results})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predictions")
@admin_required
def get_predictions():
    conn = get_db()
    rows = fetchall(
        execute(
            conn,
            """
            SELECT p.id, p.user_id, u.email, p.disease, p.probability, p.risk, p.confidence, p.created_at
            FROM predictions p
            LEFT JOIN users u ON u.id = p.user_id
            ORDER BY p.created_at DESC
            """
        )
    )
    conn.close()
    return jsonify([normalize_prediction_row(row) for row in rows])


@app.route("/api/report/<int:prediction_id>")
@login_required
def report(prediction_id):
    conn = get_db()
    prediction = fetchone(execute(conn, "SELECT * FROM predictions WHERE id=?", (prediction_id,)))
    conn.close()

    if not prediction:
        return jsonify({"error": "Prediction not found"}), 404
    if current_user.role != "admin" and prediction["user_id"] != int(current_user.id):
        return jsonify({"error": "Access denied"}), 403

    path = generate_report(normalize_prediction_row(prediction))
    return send_file(path, as_attachment=True)


@app.route("/api/email-report/<int:prediction_id>", methods=["POST"])
@login_required
def email_report(prediction_id):
    conn = get_db()
    prediction = fetchone(
        execute(
            conn,
            "SELECT p.*, u.email FROM predictions p LEFT JOIN users u ON u.id=p.user_id WHERE p.id=?",
            (prediction_id,),
        )
    )
    conn.close()

    if not prediction:
        return jsonify({"error": "Prediction not found"}), 404
    if current_user.role != "admin" and prediction["user_id"] != int(current_user.id):
        return jsonify({"error": "Access denied"}), 403

    prediction = normalize_prediction_row(prediction)
    report_path = generate_report(prediction)
    body = (
        "Health AI prediction summary\n"
        f"Disease: {prediction['disease'].title()}\n"
        f"Risk: {prediction['risk']}\n"
        f"Probability: {prediction['probability'] * 100:.2f}%\n"
        f"Confidence: {prediction['confidence'] * 100:.2f}%\n"
        "This is an AI screening summary, not a medical diagnosis."
    )
    send_email(
        prediction.get("email") or current_user.email,
        "Health AI Prediction Report",
        body,
        attachments=[{
            "path": report_path,
            "filename": f"health_report_{prediction_id}.pdf",
            "maintype": "application",
            "subtype": "pdf",
        }],
    )
    return jsonify({"message": "Report email sent."})


@app.route("/api/retrain", methods=["POST"])
@admin_required
def retrain():
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "src", "train.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr or result.stdout}), 500
    return jsonify({"message": "Models retrained successfully", "output": result.stdout})


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
