import joblib

from src.feature_catalog import enrich_feature_schema
from src.preprocess import transform_input
from src.shap_explain import get_shap_values
from src.train import MODEL_CONFIG


def risk_label(probability):
    if probability < 0.3:
        return "Low"
    if probability < 0.6:
        return "Medium"
    return "High"


def confidence_score(probability):
    return round(float(max(probability, 1 - probability)), 4)


def load_artifacts(disease):
    if disease not in MODEL_CONFIG:
        raise ValueError(f"Unsupported disease: {disease}")

    config = MODEL_CONFIG[disease]
    return {
        "model": joblib.load(config["model_path"]),
        "preprocessor": joblib.load(config["scaler_path"]),
        "features": enrich_feature_schema(disease, joblib.load(config["features_path"])),
    }


def predict_with_explain(data, disease="diabetes"):
    artifacts = load_artifacts(disease)
    expected_count = len(artifacts["features"])

    if len(data) != expected_count:
        raise ValueError(
            f"{disease.title()} prediction requires {expected_count} features."
        )

    input_data = transform_input(data, artifacts["preprocessor"], artifacts["features"])
    probability = float(artifacts["model"].predict_proba(input_data)[0][1])

    try:
        explanation = get_shap_values(
            artifacts["model"],
            input_data,
            artifacts["features"],
            artifacts["preprocessor"],
        )
    except Exception:
        explanation = {}

    return probability, risk_label(probability), confidence_score(probability), explanation
