import numpy as np
import shap


def _transformed_feature_names(preprocessor):
    names = preprocessor.get_feature_names_out()
    return [str(name) for name in names]


def _base_feature_name(transformed_name, feature_schema):
    clean_name = transformed_name.split("__", 1)[-1]
    candidates = sorted((item["name"] for item in feature_schema), key=len, reverse=True)
    for candidate in candidates:
        if clean_name == candidate or clean_name.startswith(f"{candidate}_"):
            return candidate
    return clean_name


def get_shap_values(model, input_data, feature_schema, preprocessor, top_n=8):
    transformed_names = _transformed_feature_names(preprocessor)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    values = np.asarray(shap_values)[0]

    aggregate = {}
    label_lookup = {item["name"]: item.get("label", item["name"]) for item in feature_schema}
    for feature_name, value in zip(transformed_names, values):
        base_name = _base_feature_name(feature_name, feature_schema)
        aggregate[base_name] = aggregate.get(base_name, 0.0) + float(value)

    sorted_items = sorted(aggregate.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
    return {
        label_lookup.get(name, name): round(score, 4)
        for name, score in sorted_items
    }
