from chatbot.ai_service import ask_gemini


SYMPTOM_FLOW = [
    ("fever", "Do you have a fever?"),
    ("cough", "Do you have a cough?"),
    ("chest_pain", "Any chest pain or tightness?"),
    ("breathlessness", "Any shortness of breath?"),
    ("fatigue", "Are you feeling unusual fatigue?"),
]


def _is_yes(text):
    return text.strip().lower() in {"yes", "y", "yeah", "true", "1", "have"}


def _start_state():
    return {"mode": "symptoms", "step": 0, "answers": {}}


def get_bot_response(text, state=None):
    state = state or {}
    message = text.strip().lower()

    if message in {"reset", "restart", "start over"}:
        state = {}

    if not state or message in {"symptoms", "check symptoms", "start"}:
        state = _start_state()
        return {
            "reply": "I can ask a few symptom questions step by step. " + SYMPTOM_FLOW[0][1],
            "state": state,
            "done": False,
        }

    if state.get("mode") == "symptoms":
        step = int(state.get("step", 0))
        if step < len(SYMPTOM_FLOW):
            key, _question = SYMPTOM_FLOW[step]
            state.setdefault("answers", {})[key] = _is_yes(message)
            step += 1
            state["step"] = step

        if step < len(SYMPTOM_FLOW):
            return {
                "reply": SYMPTOM_FLOW[step][1],
                "state": state,
                "done": False,
            }

        positives = [key.replace("_", " ") for key, yes in state["answers"].items() if yes]
        risk = "High" if {"chest_pain", "breathlessness"} & set(k for k, v in state["answers"].items() if v) else "Medium" if len(positives) >= 2 else "Low"
        advice = "Please seek urgent medical care." if risk == "High" else "Consider booking a routine consultation." if risk == "Medium" else "Monitor symptoms and maintain healthy habits."
        state["mode"] = "complete"
        return {
            "reply": f"Symptom check complete. Reported symptoms: {', '.join(positives) or 'none'}. Initial risk: {risk}. {advice}",
            "state": state,
            "done": True,
        }

    if "diabetes" in message:
        reply = "Diabetes risk is influenced by glucose, BMI, age, and blood pressure. You can check it from the dashboard."
    elif "heart" in message:
        reply = "Heart risk screening uses age, sex, chest pain type, resting blood pressure, cholesterol, and max heart rate."
    elif "diet" in message:
        reply = "Prefer high-fiber meals, enough protein, less added sugar, and fewer fried foods."
    else:
        reply = "Type 'start' for a step-by-step symptom check, or ask about diabetes, heart risk, diet, or exercise."

    return {"reply": reply, "state": state, "done": False}


def get_smart_bot_response(text, history=None, state=None):
    ai_reply = ask_gemini(text.strip(), history or [])
    if ai_reply:
        return {
            "reply": ai_reply,
            "state": {"mode": "ai"},
            "done": False,
            "source": "gemini",
        }

    fallback = get_bot_response(text, state=state)
    fallback["source"] = "rule-based"
    return fallback
