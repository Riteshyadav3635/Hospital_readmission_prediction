from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

FEATURE_DEFAULTS = {
    "race": 2,
    "gender": 0,
    "age": 7,
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 4,
    "medical_specialty": 71,
    "num_lab_procedures": 43,
    "num_procedures": 1,
    "num_medications": 16,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": 276,
    "diag_2": 133,
    "diag_3": 86,
    "number_diagnoses": 8,
    "metformin": 0,
    "repaglinide": 0,
    "nateglinide": 0,
    "chlorpropamide": 0,
    "glimepiride": 0,
    "acetohexamide": 0,
    "glipizide": 0,
    "glyburide": 0,
    "tolbutamide": 0,
    "pioglitazone": 0,
    "rosiglitazone": 0,
    "acarbose": 0,
    "miglitol": 0,
    "troglitazone": 0,
    "tolazamide": 0,
    "insulin": 0,
    "glyburide-metformin": 0,
    "glipizide-metformin": 0,
    "glimepiride-pioglitazone": 0,
    "metformin-rosiglitazone": 0,
    "metformin-pioglitazone": 0,
    "change": 1,
    "diabetesMed": 1,
}

AGE_MAPPING = {
    "[0-10)": 0,
    "[10-20)": 1,
    "[20-30)": 2,
    "[30-40)": 3,
    "[40-50)": 4,
    "[50-60)": 5,
    "[60-70)": 6,
    "[70-80)": 7,
    "[80-90)": 8,
    "[90-100)": 9,
}

GENDER_MAPPING = {
    "Female": 0,
    "Male": 1,
    "Unknown/Invalid": 2,
}

MEDICATION_MAPPING = {
    "No": 0,
    "Steady": 1,
    "Up": 2,
    "Down": 3,
}

BINARY_MAPPING = {
    "No": 0,
    "Yes": 1,
}

CHANGE_MAPPING = {
    "Ch": 0,
    "No": 1,
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "ml" / "models"


def get_feature_columns(scaler):
    feature_columns = getattr(scaler, "feature_names_in_", None)
    if feature_columns is None:
        raise ValueError("Scaler does not contain feature_names_in_.")
    return list(feature_columns)


def load_label_encoders(model_dir=MODEL_DIR):
    with open(model_dir / "label_encoders.pkl", "rb") as file:
        return pickle.load(file)


def load_artifacts(model_dir=MODEL_DIR):
    with open(model_dir / "scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    with open(model_dir / "pca_95.pkl", "rb") as file:
        pca = pickle.load(file)
    with open(model_dir / "hospital_readmission_model.pkl", "rb") as file:
        model = pickle.load(file)
    validate_artifacts(scaler, pca, model)
    return scaler, pca, model


def validate_artifacts(scaler, pca, model):
    feature_columns = get_feature_columns(scaler)
    scaler_features = len(feature_columns)
    pca_input_features = getattr(pca, "n_features_in_", None)
    model_input_features = getattr(model, "n_features_in_", None)
    pca_components = getattr(pca, "n_components_", None)

    if pca_input_features != scaler_features:
        raise ValueError(f"PCA expects {pca_input_features} features, but scaler has {scaler_features}.")

    if model_input_features != pca_components:
        raise ValueError(f"Model expects {model_input_features} features, but PCA outputs {pca_components}.")

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for probability output.")


def build_feature_frame(values: dict[str, Any], feature_columns):
    row = {column: FEATURE_DEFAULTS.get(column, 0) for column in feature_columns}
    unknown_columns = sorted(set(values) - set(feature_columns))
    if unknown_columns:
        raise ValueError(f"Unknown feature columns: {', '.join(unknown_columns)}")
    row.update(values)
    return pd.DataFrame([row], columns=feature_columns)


def encode_with_label_encoder(value: Any, encoder, fallback: Any):
    text_value = str(value).strip()
    if text_value in encoder.classes_:
        return int(encoder.transform([text_value])[0])
    return fallback


def normalize_enum(value: Any, mapping: dict[str, int], fallback: int):
    if value is None:
        return fallback
    key = str(value).strip()
    if key in mapping:
        return mapping[key]
    if isinstance(value, (int, float)):
        return int(value)
    return fallback


def prepare_model_input(raw_input: dict[str, Any]) -> dict[str, Any]:
    payload = FEATURE_DEFAULTS.copy()
    if not raw_input:
        return payload

    input_data = {}
    alias_map = {
        "age_group": "age",
        "age-group": "age",
        "medical_specialty": "medical_specialty",
        "medical-specialty": "medical_specialty",
        "diag_1": "diag_1",
        "diag-1": "diag_1",
        "diag_2": "diag_2",
        "diag-2": "diag_2",
        "diag_3": "diag_3",
        "diag-3": "diag_3",
        "glyburide_metformin": "glyburide-metformin",
        "glyburide-metformin": "glyburide-metformin",
        "glipizide_metformin": "glipizide-metformin",
        "glipizide-metformin": "glipizide-metformin",
        "glimepiride_pioglitazone": "glimepiride-pioglitazone",
        "glimepiride-pioglitazone": "glimepiride-pioglitazone",
        "metformin_rosiglitazone": "metformin-rosiglitazone",
        "metformin-rosiglitazone": "metformin-rosiglitazone",
        "metformin_pioglitazone": "metformin-pioglitazone",
        "metformin-pioglitazone": "metformin-pioglitazone",
    }

    for key, value in dict(raw_input).items():
        if key is None:
            continue
        canonical = alias_map.get(str(key), str(key).replace("_", "-"))
        input_data[canonical] = value

    if "age" in input_data and input_data["age"] is not None and isinstance(input_data["age"], str):
        input_data["age"] = AGE_MAPPING.get(input_data["age"], FEATURE_DEFAULTS["age"])

    if "age" not in input_data and "age_group" in input_data:
        input_data["age"] = AGE_MAPPING.get(input_data["age_group"], FEATURE_DEFAULTS["age"])

    if "age" in input_data and input_data["age"] is not None:
        payload["age"] = int(input_data["age"])

    for key in [
        "race",
        "gender",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
    ]:
        if key in input_data and input_data[key] is not None:
            value = input_data[key]
            if isinstance(value, (int, float)):
                payload[key] = int(value)
                continue
            encoders = load_label_encoders()
            if key in encoders:
                payload[key] = encode_with_label_encoder(value, encoders[key], FEATURE_DEFAULTS.get(key, 0))
            else:
                payload[key] = FEATURE_DEFAULTS.get(key, 0)

    for key in [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]:
        if key in input_data and input_data[key] is not None:
            payload[key] = normalize_enum(input_data[key], MEDICATION_MAPPING, FEATURE_DEFAULTS.get(key, 0))

    for key in ["change", "diabetesMed"]:
        if key in input_data and input_data[key] is not None:
            mapping = CHANGE_MAPPING if key == "change" else BINARY_MAPPING
            payload[key] = normalize_enum(input_data[key], mapping, FEATURE_DEFAULTS.get(key, 1 if key == "change" else 1))

    for key, value in input_data.items():
        if key in payload and value is not None and key not in {"age", "race", "gender", "medical_specialty", "diag_1", "diag_2", "diag_3", "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed"}:
            payload[key] = value

    return payload


@lru_cache(maxsize=1)
def get_model_bundle():
    scaler, pca, model = load_artifacts()
    encoders = load_label_encoders()
    return {"scaler": scaler, "pca": pca, "model": model, "encoders": encoders}


def predict_payload(input_data: dict[str, Any]):
    bundle = get_model_bundle()
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    model = bundle["model"]

    feature_columns = get_feature_columns(scaler)
    prepared = prepare_model_input(input_data)
    frame = build_feature_frame(prepared, feature_columns)

    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Missing feature values for: {', '.join(missing)}")

    scaled = scaler.transform(frame)
    pca_features = pca.transform(scaled)
    probability = float(model.predict_proba(pca_features)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": "High Risk of Readmission" if prediction == 1 else "Low Risk of Readmission",
        "model_features": prepared,
    }
