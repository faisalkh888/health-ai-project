import json
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _percent(value):
    return f"{float(value) * 100:.2f}%"


def _feature_rows(data):
    values = data.get("feature_values")
    if not values:
        return []
    if isinstance(values, str):
        values = json.loads(values)
    return [[key, value] for key, value in values.items()]


def generate_report(data):
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, f"report_{data['id']}.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Health AI Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))

    summary = [
        ["Disease", str(data.get("disease", "")).title()],
        ["Risk Level", data.get("risk", "N/A")],
        ["Probability", _percent(data.get("probability", 0))],
        ["Confidence", _percent(data.get("confidence", 0))],
        ["Created", str(data.get("created_at", "N/A"))],
    ]
    table = Table(summary, hAlign="LEFT", colWidths=[120, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    feature_rows = _feature_rows(data)
    if feature_rows:
        story.append(Paragraph("Input Data", styles["Heading2"]))
        feature_table = Table([["Feature", "Value"]] + feature_rows, hAlign="LEFT")
        feature_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(feature_table)
        story.append(Spacer(1, 16))

    story.append(Paragraph("Guidance", styles["Heading2"]))
    story.append(Paragraph(
        "This report is an AI screening summary, not a medical diagnosis. "
        "Please consult a qualified clinician for medical decisions.",
        styles["BodyText"],
    ))

    doc.build(story)
    return path
