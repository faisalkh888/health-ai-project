import math

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def normalize_yes_no(value):
    if pd.isna(value):
        return value
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true", "present", "positive"}:
        return 1
    if text in {"no", "n", "0", "false", "notpresent", "negative"}:
        return 0
    return value


def normalize_binary_target(series):
    cleaned = series.apply(normalize_yes_no)
    numeric_cleaned = pd.to_numeric(cleaned, errors="coerce")
    unique_numeric = set(numeric_cleaned.dropna().astype(int).unique())
    if unique_numeric <= {1, 2} and unique_numeric:
        return numeric_cleaned.map({1: 1, 2: 0}).fillna(0).astype(int)
    if cleaned.dtype == object:
        lowered = cleaned.astype(str).str.strip().str.lower()
        unique_values = set(lowered.dropna().unique())
        if unique_values <= {"1", "2"}:
            cleaned = lowered.map({"1": 1, "2": 0})
        elif unique_values <= {"ckd", "notckd"}:
            cleaned = lowered.map({"ckd": 1, "notckd": 0})
    return pd.to_numeric(cleaned, errors="coerce").fillna(0).astype(int)


def clean_dataframe(df):
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    cleaned = cleaned.replace("?", pd.NA)
    for column in cleaned.columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.decode("utf-8").strip() if isinstance(value, bytes) else value
        )
        cleaned[column] = cleaned[column].map(normalize_yes_no)
    return cleaned


def build_feature_schema(df, selected_features):
    schema = []
    for feature in selected_features:
        series = df[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(numeric.notna().mean()) if len(series) else 0.0
        if numeric_ratio >= 0.8:
            schema.append({"name": feature, "kind": "number", "label": feature})
            continue

        options = [str(value) for value in sorted(series.dropna().astype(str).unique().tolist())]
        schema.append({
            "name": feature,
            "kind": "select",
            "label": feature,
            "options": options,
        })
    return schema


def build_preprocessor(df, schema):
    numeric_features = [item["name"] for item in schema if item["kind"] == "number"]
    categorical_features = [item["name"] for item in schema if item["kind"] == "select"]

    transformers = []
    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))
    return ColumnTransformer(transformers=transformers)


def preprocess_data(df, target_column, selected_features=None):
    cleaned = clean_dataframe(df)
    selected_features = selected_features or [column for column in cleaned.columns if column != target_column]
    X = cleaned[selected_features].copy()
    y = normalize_binary_target(cleaned[target_column])
    schema = build_feature_schema(X, selected_features)
    for item in schema:
        if item["kind"] == "number":
            X[item["name"]] = pd.to_numeric(X[item["name"]], errors="coerce")
    preprocessor = build_preprocessor(X, schema)
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, schema, preprocessor


def coerce_feature_value(value, spec):
    if spec["kind"] == "number":
        if value in (None, ""):
            return math.nan
        return float(value)
    if value is None:
        return ""
    return str(value).strip()


def transform_input(data, preprocessor, feature_schema):
    row = {
        spec["name"]: coerce_feature_value(value, spec)
        for spec, value in zip(feature_schema, data)
    }
    frame = pd.DataFrame([row], columns=[spec["name"] for spec in feature_schema])
    return preprocessor.transform(frame)
