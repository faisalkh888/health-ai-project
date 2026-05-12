import sqlite3
import struct
from io import BytesIO

import pytest
from reportlab.pdfgen import canvas
from werkzeug.security import generate_password_hash

import app as health_app
import auth.auth_routes as auth_routes


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(health_app, "DATABASE", str(db_path))
    monkeypatch.setattr(auth_routes, "DATABASE", str(db_path))
    health_app.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    health_app.init_db()

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
        ("Patient", "patient@example.com", generate_password_hash("pass"), "patient"),
    )
    conn.execute(
        "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
        ("Admin", "admin@example.com", generate_password_hash("pass"), "admin"),
    )
    conn.commit()
    conn.close()

    with health_app.app.test_client() as test_client:
        yield test_client


def login(client, user_id):
    with client.session_transaction() as session:
        session["_user_id"] = str(user_id)
        session["_fresh"] = True


def test_dashboard_requires_login(client):
    response = client.get("/dashboard")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_role_login_routes_redirect_to_correct_dashboard(client):
    patient = client.post(
        "/patient/login",
        data={"email": "patient@example.com", "password": "pass"},
    )
    assert patient.status_code == 302
    assert "/patient/dashboard" in patient.headers["Location"]

    client.get("/logout")

    admin = client.post(
        "/admin/login",
        data={"email": "admin@example.com", "password": "pass"},
    )
    assert admin.status_code == 302
    assert "/admin/dashboard" in admin.headers["Location"]


def test_admin_login_is_case_insensitive(client):
    response = client.post(
        "/admin/login",
        data={"email": "ADMIN@EXAMPLE.COM", "password": "pass"},
    )
    assert response.status_code == 302
    assert "/admin/dashboard" in response.headers["Location"]


def test_admin_login_rejects_patient_account(client):
    response = client.post(
        "/admin/login",
        data={"email": "patient@example.com", "password": "pass"},
    )
    assert response.status_code == 403
    assert b"patient sign in page" in response.data


def test_patient_cannot_open_admin(client):
    login(client, 1)
    response = client.get("/admin")
    assert response.status_code == 302
    assert "/patient/dashboard" in response.headers["Location"]

    api_response = client.get("/api/predictions")
    assert api_response.status_code == 403


def test_admin_can_list_predictions(client):
    login(client, 2)
    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 302
    assert "/admin/dashboard" in dashboard.headers["Location"]

    response = client.get("/api/predictions")
    assert response.status_code == 200
    assert response.get_json() == []


def test_admin_dashboard_handles_old_byte_probability(client):
    conn = sqlite3.connect(health_app.DATABASE)
    conn.execute(
        """
        INSERT INTO predictions (user_id, disease, probability, risk)
        VALUES (?, ?, ?, ?)
        """,
        (1, "diabetes", struct.pack("<f", 0.25), "Low"),
    )
    conn.commit()
    conn.close()

    login(client, 2)
    response = client.get("/admin/dashboard")
    assert response.status_code == 200
    assert b"25.00%" in response.data


def test_patient_dashboard_loads(client):
    login(client, 1)
    response = client.get("/patient/dashboard")
    assert response.status_code == 200
    assert b"Patient Dashboard" in response.data


def test_prediction_validation(client):
    login(client, 1)
    response = client.post("/api/predict/diabetes", json={"features": [1, 2]})
    assert response.status_code == 400
    assert "requires 4 features" in response.get_json()["error"]


def test_prediction_api_does_not_require_csrf(client, monkeypatch):
    login(client, 1)
    monkeypatch.setattr(
        health_app,
        "predict_with_explain",
        lambda features, disease: (0.12, "Low", 0.88, {"Glucose": -0.3}),
    )
    response = client.post(
        "/api/predict/diabetes",
        json={"features": [1, 3, 5, 9]},
    )
    assert response.status_code == 200
    assert response.get_json()["risk"] == "Low"


def test_prediction_is_saved(client, monkeypatch):
    login(client, 1)
    monkeypatch.setattr(
        health_app,
        "predict_with_explain",
        lambda features, disease: (0.72, "High", 0.72, {"Glucose": 0.5}),
    )

    response = client.post(
        "/api/predict/diabetes",
        json={"features": [85, 26.6, 31, 66]},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["id"] == 1
    assert data["risk"] == "High"
    assert data["confidence"] == 0.72

    admin_response = client.get("/api/predictions")
    assert admin_response.status_code == 403

    login(client, 2)
    admin_response = client.get("/api/predictions")
    assert admin_response.status_code == 200
    rows = admin_response.get_json()
    assert rows[0]["disease"] == "diabetes"


def test_pdf_upload_prediction(client, monkeypatch):
    login(client, 1)
    monkeypatch.setattr(
        health_app,
        "predict_with_explain",
        lambda features, disease: (0.2, "Low", 0.8, {"Glucose": -0.2}),
    )

    pdf = BytesIO()
    doc = canvas.Canvas(pdf)
    doc.drawString(100, 750, "Glucose: 85")
    doc.drawString(100, 730, "BMI: 26.6")
    doc.drawString(100, 710, "Age: 31")
    doc.drawString(100, 690, "BloodPressure: 66")
    doc.save()
    pdf.seek(0)

    response = client.post(
        "/api/predict/diabetes/upload",
        data={"file": (pdf, "diabetes.pdf")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["disease"] == "diabetes"
