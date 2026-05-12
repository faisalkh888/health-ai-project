import csv
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import preprocess_data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


MODEL_CONFIG = {
    "diabetes": {
        "version": "diabetes-v2",
        "data_path": os.path.join(PROJECT_ROOT, "data", "diabetes.csv"),
        "target": "Outcome",
        "features": ["Glucose", "BMI", "Age", "BloodPressure"],
    },
    "heart": {
        "version": "heart-v2",
        "data_path": os.path.join(PROJECT_ROOT, "data", "heart.csv"),
        "target": "target",
        "features": ["age", "sex", "cp", "trestbps", "chol", "thalach"],
    },
    "cancer": {
        "version": "cancer-v1",
        "loader": "breast_cancer",
        "target": "target",
        "features": [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean smoothness",
            "worst concavity",
        ],
    },
    "kidney": {
        "version": "kidney-v1",
        "data_path": os.path.join(PROJECT_ROOT, "data", "kidney.arff"),
        "loader": "kidney_arff",
        "target": "class",
        "features": ["age", "bp", "sg", "al", "sc", "hemo", "htn", "dm"],
    },
    "liver": {
        "version": "liver-v1",
        "data_path": os.path.join(PROJECT_ROOT, "data", "liver.csv"),
        "loader": "liver_csv",
        "target": "dataset",
        "features": [
            "age",
            "gender",
            "total_bilirubin",
            "alkphos",
            "sgpt",
            "sgot",
            "albumin",
            "a_g_ratio",
        ],
    },
    "stroke": {
        "version": "stroke-v1",
        "data_path": os.path.join(PROJECT_ROOT, "data", "stroke.csv"),
        "target": "stroke",
        "features": [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "avg_glucose_level",
            "bmi",
            "smoking_status",
        ],
    },
    "mental_health": {
        "version": "mental-health-v1",
        "data_path": os.path.join(PROJECT_ROOT, "data", "mental_health.csv"),
        "loader": "mental_health_csv",
        "target": "Do you have Depression?",
        "features": [
            "Choose your gender",
            "Age",
            "Your current year of Study",
            "What is your CGPA?",
            "Marital status",
            "Do you have Anxiety?",
            "Do you have Panic attack?",
            "Did you seek any specialist for a treatment?",
        ],
    },
}


for disease, config in MODEL_CONFIG.items():
    config["model_path"] = os.path.join(PROJECT_ROOT, "models", f"{disease}_model.pkl")
    config["scaler_path"] = os.path.join(PROJECT_ROOT, "models", f"{disease}_scaler.pkl")
    config["features_path"] = os.path.join(PROJECT_ROOT, "models", f"{disease}_features.pkl")
    config["metrics_path"] = os.path.join(PROJECT_ROOT, "models", f"{disease}_metrics.json")


def load_liver_dataset(path):
    columns = [
        "age",
        "gender",
        "total_bilirubin",
        "direct_bilirubin",
        "alkphos",
        "sgpt",
        "sgot",
        "total_proteins",
        "albumin",
        "a_g_ratio",
        "dataset",
    ]
    return pd.read_csv(path, names=columns, header=None)


def load_kidney_dataset(path):
    attributes = []
    data_rows = []
    in_data = False
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            lower = line.lower()
            if lower.startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    attributes.append(parts[1].strip("'\""))
            elif lower.startswith("@data"):
                in_data = True
            elif in_data:
                row = next(csv.reader([line]))
                if len(row) == len(attributes) + 1:
                    if row[-1] == "":
                        row = row[:-1]
                    elif "" in row:
                        row.remove("")
                if len(row) != len(attributes):
                    continue
                data_rows.append(row)
    return pd.DataFrame(data_rows, columns=attributes)


def load_cancer_dataset():
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame["target"] = dataset.target
    return frame


def load_mental_health_dataset(path):
    df = pd.read_csv(path)
    return df.drop(columns=["Timestamp", "What is your course?"], errors="ignore")


def load_dataset(config):
    loader = config.get("loader")
    if loader == "liver_csv":
        return load_liver_dataset(config["data_path"])
    if loader == "kidney_arff":
        return load_kidney_dataset(config["data_path"])
    if loader == "breast_cancer":
        return load_cancer_dataset()
    if loader == "mental_health_csv":
        return load_mental_health_dataset(config["data_path"])
    return pd.read_csv(config["data_path"])


def build_model(scale_pos_weight=1.0):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_estimators=160,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
    )


def tune_model(X_train, y_train):
    positives = max(int((y_train == 1).sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    base_weight = round(negatives / positives, 2)

    estimator = build_model(scale_pos_weight=base_weight)
    search = GridSearchCV(
        estimator,
        param_grid={
            "n_estimators": [120, 180],
            "max_depth": [3, 5],
            "learning_rate": [0.05],
        },
        scoring="roc_auc",
        cv=3,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_disease(disease):
    config = MODEL_CONFIG[disease]
    df = load_dataset(config)
    X, y, feature_schema, preprocessor = preprocess_data(
        df, config["target"], config.get("features")
    )

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model, best_params = tune_model(X_train, y_train)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "disease": disease,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_size": int(len(y_test)),
        "features": [item["name"] for item in feature_schema],
        "feature_schema": feature_schema,
        "best_params": best_params,
    }

    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    joblib.dump(model, config["model_path"])
    joblib.dump(preprocessor, config["scaler_path"])
    joblib.dump(feature_schema, config["features_path"])

    with open(config["metrics_path"], "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


def train():
    results = {}
    for disease in MODEL_CONFIG:
        results[disease] = train_disease(disease)
        print(f"{disease.title()} model trained: {results[disease]}")
    return results


if __name__ == "__main__":
    train()
