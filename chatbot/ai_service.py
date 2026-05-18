import json
from urllib import error, request

from config import GEMINI_API_KEY, GEMINI_MODEL


SYSTEM_PROMPT = (
    "You are Health AI, a careful medical assistant for a screening web application. "
    "You can explain prediction reports, ask short follow-up questions, summarize likely next steps, "
    "and encourage professional medical care when symptoms are serious. "
    "Do not claim to diagnose. Keep responses concise, empathetic, and practical. "
    "Always mention that urgent symptoms like chest pain, fainting, stroke signs, or severe breathing difficulty "
    "need immediate in-person care."
)


def gemini_enabled():
    return bool(GEMINI_API_KEY)


def build_contents(history, message):
    contents = []
    for item in history[-8:]:
        role = "model" if item.get("role") == "assistant" else "user"
        text = (item.get("message") or "").strip()
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})
    contents.append({"role": "user", "parts": [{"text": message}]})
    return contents


def ask_gemini(message, history):
    if not gemini_enabled():
        return None

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    payload = json.dumps({
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}],
        },
        "contents": build_contents(history, message),
        "generationConfig": {
            "temperature": 0.3,
            "topP": 0.9,
            "maxOutputTokens": 350,
        },
    }).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    candidates = data.get("candidates") or []
    if not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts", [])
    text = " ".join(part.get("text", "").strip() for part in parts).strip()
    return text or None
