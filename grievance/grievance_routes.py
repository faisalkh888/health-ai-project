import sqlite3

from flask import Blueprint, flash, redirect, render_template, request
from flask_login import current_user, login_required

from config import DATABASE

grievance_bp = Blueprint("grievance", __name__)


def get_db():
    return sqlite3.connect(DATABASE)


@grievance_bp.route("/grievance", methods=["GET", "POST"])
@login_required
def grievance():
    if request.method == "POST":
        message = request.form["message"]

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO grievances (user_id, message) VALUES (?, ?)",
            (current_user.id, message),
        )
        conn.commit()
        conn.close()

        flash("Complaint submitted successfully!")
        return redirect("/grievance")

    return render_template("grievance.html")
