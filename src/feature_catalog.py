FEATURE_METADATA = {
    "diabetes": {
        "Glucose": {"label": "Glucose", "placeholder": "mg/dL"},
        "BMI": {"label": "Body Mass Index", "placeholder": "e.g. 26.6"},
        "Age": {"label": "Age", "placeholder": "years"},
        "BloodPressure": {"label": "Blood Pressure", "placeholder": "mm/Hg"},
    },
    "heart": {
        "age": {"label": "Age", "placeholder": "years"},
        "sex": {"label": "Sex", "placeholder": "0 = Female, 1 = Male"},
        "cp": {"label": "Chest Pain Type", "placeholder": "0-3"},
        "trestbps": {"label": "Resting Blood Pressure", "placeholder": "mm/Hg"},
        "chol": {"label": "Cholesterol", "placeholder": "mg/dL"},
        "thalach": {"label": "Max Heart Rate", "placeholder": "bpm"},
    },
    "cancer": {
        "mean radius": {"label": "Mean Radius", "placeholder": "tumor radius"},
        "mean texture": {"label": "Mean Texture", "placeholder": "texture score"},
        "mean perimeter": {"label": "Mean Perimeter", "placeholder": "perimeter"},
        "mean area": {"label": "Mean Area", "placeholder": "area"},
        "mean smoothness": {"label": "Mean Smoothness", "placeholder": "smoothness"},
        "worst concavity": {"label": "Worst Concavity", "placeholder": "concavity"},
    },
    "kidney": {
        "age": {"label": "Patient Age", "placeholder": "years"},
        "bp": {"label": "Blood Pressure", "placeholder": "mm/Hg"},
        "sg": {"label": "Specific Gravity", "placeholder": "1.005-1.025"},
        "al": {"label": "Albumin", "placeholder": "0-5"},
        "sc": {"label": "Serum Creatinine", "placeholder": "mg/dL"},
        "hemo": {"label": "Hemoglobin", "placeholder": "g/dL"},
        "htn": {"label": "Hypertension", "placeholder": "0 = No, 1 = Yes"},
        "dm": {"label": "Diabetes Mellitus", "placeholder": "0 = No, 1 = Yes"},
    },
    "liver": {
        "age": {"label": "Age", "placeholder": "years"},
        "gender": {"label": "Gender"},
        "total_bilirubin": {"label": "Total Bilirubin", "placeholder": "mg/dL"},
        "alkphos": {"label": "Alkaline Phosphotase", "placeholder": "IU/L"},
        "sgpt": {"label": "SGPT / ALT", "placeholder": "IU/L"},
        "sgot": {"label": "SGOT / AST", "placeholder": "IU/L"},
        "albumin": {"label": "Albumin", "placeholder": "g/dL"},
        "a_g_ratio": {"label": "Albumin/Globulin Ratio", "placeholder": "ratio"},
    },
    "stroke": {
        "gender": {"label": "Gender"},
        "age": {"label": "Age", "placeholder": "years"},
        "hypertension": {"label": "Hypertension", "placeholder": "0 = No, 1 = Yes"},
        "heart_disease": {"label": "Heart Disease", "placeholder": "0 = No, 1 = Yes"},
        "ever_married": {"label": "Ever Married", "placeholder": "0 = No, 1 = Yes"},
        "avg_glucose_level": {"label": "Average Glucose Level", "placeholder": "mg/dL"},
        "bmi": {"label": "Body Mass Index", "placeholder": "e.g. 28.1"},
        "smoking_status": {"label": "Smoking Status"},
    },
    "mental_health": {
        "Choose your gender": {"label": "Gender"},
        "Age": {"label": "Age", "placeholder": "years"},
        "Your current year of Study": {"label": "Year of Study"},
        "What is your CGPA?": {"label": "CGPA Range"},
        "Marital status": {"label": "Marital Status", "placeholder": "0 = No, 1 = Yes"},
        "Do you have Anxiety?": {"label": "Anxiety", "placeholder": "0 = No, 1 = Yes"},
        "Do you have Panic attack?": {"label": "Panic Attack", "placeholder": "0 = No, 1 = Yes"},
        "Did you seek any specialist for a treatment?": {"label": "Specialist Support", "placeholder": "0 = No, 1 = Yes"},
    },
}


def enrich_feature_schema(disease, schema):
    metadata = FEATURE_METADATA.get(disease, {})
    enriched = []
    for item in schema:
        info = metadata.get(item["name"], {})
        enriched.append({
            **item,
            "label": info.get("label", item.get("label", item["name"])),
            "placeholder": info.get("placeholder", ""),
        })
    return enriched


def get_feature_label(disease, feature_name):
    return FEATURE_METADATA.get(disease, {}).get(feature_name, {}).get("label", feature_name)
