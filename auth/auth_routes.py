# auth/auth_routes.py
from flask import Blueprint, flash, request, render_template, redirect
import secrets
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_required, login_user, logout_user
from auth.user_model import User
from config import DATABASE as CONFIG_DATABASE
from src.db import connect, execute, fetchone, fetch_value
from src.email_utils import send_email

auth_bp = Blueprint("auth", __name__)
DATABASE = CONFIG_DATABASE

def get_db():
    if DATABASE != CONFIG_DATABASE:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        return conn
    return connect()


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    return role_login(None)


@auth_bp.route("/patient/login", methods=["GET", "POST"])
def patient_login():
    return role_login("patient")


@auth_bp.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    return role_login("admin")


def role_login(required_role):
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        conn = get_db()
        user = fetchone(execute(conn, "SELECT * FROM users WHERE lower(email)=?", (email,)))
        conn.close()

        if user and check_password_hash(user["password"], password):
            if required_role and user["role"] != required_role:
                flash(f"Please use the {user['role']} sign in page for this account.")
                return render_template("login.html", role=required_role), 403

            user_obj = User(user["id"], user["name"], user["email"], user["role"])
            login_user(user_obj)
            if user["role"] == "admin":
                return redirect("/admin/dashboard")
            return redirect("/patient/dashboard")

        flash("Invalid email or password.")

    return render_template("login.html", role=required_role)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = generate_password_hash(request.form["password"])

        conn = get_db()
        user_count = fetch_value(conn, "SELECT COUNT(*) FROM users") or 0
        role = "admin" if user_count == 0 else "patient"

        try:
            execute(
                conn,
                "INSERT INTO users (name,email,password,role) VALUES (?,?,?,?)",
                (name, email, password, role)
            )
            conn.commit()
        except Exception as exc:
            if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                flash("An account with this email already exists.")
                return render_template("register.html"), 400
            raise
        finally:
            conn.close()

        return redirect("/login")

    return render_template("register.html")


@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        conn = get_db()
        user = fetchone(execute(conn, "SELECT id, email FROM users WHERE lower(email)=?", (email,)))
        if user:
            token = secrets.token_urlsafe(32)
            execute(
                conn,
                "INSERT INTO password_reset_tokens (user_id, token) VALUES (?, ?)",
                (user["id"], token),
            )
            conn.commit()
            link = request.host_url.rstrip("/") + f"/reset-password/{token}"
            send_email(user["email"], "Health AI Password Reset", f"Reset your password: {link}")
        conn.close()
        flash("If the email exists, a reset link has been sent.")
        return redirect("/forgot-password")
    return render_template("forgot_password.html")


@auth_bp.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    conn = get_db()
    row = fetchone(
        execute(
            conn,
            """
            SELECT id, user_id FROM password_reset_tokens
            WHERE token=? AND used=0
            """,
            (token,),
        )
    )
    if not row:
        conn.close()
        flash("Invalid or expired reset link.")
        return render_template("reset_password.html", token=token), 400

    if request.method == "POST":
        password = request.form["password"]
        execute(
            conn,
            "UPDATE users SET password=? WHERE id=?",
            (generate_password_hash(password), row["user_id"]),
        )
        execute(conn, "UPDATE password_reset_tokens SET used=1 WHERE id=?", (row["id"],))
        conn.commit()
        conn.close()
        flash("Password reset. Please sign in.")
        return redirect("/login")

    conn.close()
    return render_template("reset_password.html", token=token)


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")
