import json
import mimetypes
import re
from base64 import b64encode
from io import BytesIO
from urllib import error, request

from pypdf import PdfReader

from config import GEMINI_API_KEY
from src.feature_catalog import get_feature_label


def extract_text_from_pdf(content):
    reader = PdfReader(BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def regex_extract_features(text, feature_schema):
    values = {}
    for item in feature_schema:
        aliases = {item["name"], item.get("label", item["name"])}
        for alias in aliases:
            if item["kind"] == "number":
                pattern = rf"{re.escape(alias)}\s*[:=\-]?\s*(-?\d+(?:\.\d+)?)"
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    values[item["name"]] = float(match.group(1))
                    break
            else:
                pattern = rf"{re.escape(alias)}\s*[:=\-]?\s*([A-Za-z0-9 ._/\-]+)"
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    values[item["name"]] = match.group(1).strip()
                    break
    return values


def _gemini_payload(file_bytes, mime_type, disease, feature_schema):
    expected = [
        {
            "name": item["name"],
            "label": item.get("label", item["name"]),
            "type": item["kind"],
        }
        for item in feature_schema
    ]
    prompt = (
        f"Read this medical report and extract values needed for {disease.replace('_', ' ')} prediction. "
        "Return only valid JSON with a top-level object named features. "
        "Use the original feature names as keys. "
        "If a value is missing, do not invent it. "
        f"Expected schema: {json.dumps(expected)}"
    )
    return {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime_type, "data": b64encode(file_bytes).decode("ascii")}},
            ],
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 800,
        },
    }


def gemini_extract_features(file_bytes, mime_type, disease, feature_schema):
    if not GEMINI_API_KEY:
        return {}

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = json.dumps(_gemini_payload(file_bytes, mime_type, disease, feature_schema)).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return {}

    candidates = data.get("candidates") or []
    if not candidates:
        return {}
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        return {}

    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not json_match:
        return {}
    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed.get("features", {}) if isinstance(parsed, dict) else {}


def parse_uploaded_report(content, filename, disease, feature_schema):
    lower = filename.lower()
    text = ""
    if lower.endswith(".pdf"):
        text = extract_text_from_pdf(content)
    elif lower.endswith((".txt", ".csv")):
        text = content.decode("utf-8", errors="ignore")

    values = regex_extract_features(text, feature_schema) if text else {}
    missing = [item["name"] for item in feature_schema if item["name"] not in values]
    if not missing:
        return values

    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    ai_values = gemini_extract_features(content, mime_type, disease, feature_schema)
    for item in feature_schema:
        if item["name"] in ai_values and item["name"] not in values:
            values[item["name"]] = ai_values[item["name"]]
        elif item.get("label") in ai_values and item["name"] not in values:
            values[item["name"]] = ai_values[item["label"]]
    return values


def format_missing_fields(disease, feature_schema, values):
    missing = [get_feature_label(disease, item["name"]) for item in feature_schema if item["name"] not in values]
    return missing
