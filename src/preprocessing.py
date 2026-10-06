from __future__ import annotations

from typing import Any

import pandas as pd

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

LABEL_ENCODED_FEATURES = (
    "race",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
)

MEDICATION_FEATURES = (
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
)

DEFAULT_INPUTS: dict[str, Any] = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[50-60)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 4,
    "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 43,
    "num_procedures": 1,
    "num_medications": 16,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": "276",
    "diag_2": "133",
    "diag_3": "86",
    "number_diagnoses": 8,
    **{feature: "No" for feature in MEDICATION_FEATURES},
    "change": "No",
    "diabetesMed": "Yes",
}

LABEL_ENCODER_DEFAULT_CODES = {
    "race": 2,
    "medical_specialty": 71,
    "diag_1": 276,
    "diag_2": 133,
    "diag_3": 86,
}


def default_inputs(encoders: dict[str, Any]) -> dict[str, Any]:
    """Return the familiar example inputs, replacing non-class defaults with valid labels."""
    defaults = DEFAULT_INPUTS.copy()
    for feature in LABEL_ENCODED_FEATURES:
        options = [str(value) for value in encoders[feature].classes_]
        if str(defaults[feature]) not in options:
            fallback_index = LABEL_ENCODER_DEFAULT_CODES[feature]
            if fallback_index < len(options):
                defaults[feature] = options[fallback_index]
            else:
                defaults[feature] = options[0]
    return defaults


def encode_category(value: Any, encoder: Any, feature: str) -> int:
    """Encode one category with the exact fitted LabelEncoder classes."""
    text = str(value)
    classes = [str(label) for label in encoder.classes_]
    try:
        return classes.index(text)
    except ValueError as error:
        raise ValueError(f"Unsupported value for {feature}: {text}") from error


def prepare_features(
    input_values: dict[str, Any],
    feature_columns: list[str],
    encoders: dict[str, Any],
) -> pd.DataFrame:
    """Apply the saved project's mappings in the scaler's fitted feature order."""
    expected = set(feature_columns)
    supplied = set(input_values)
    missing = expected - supplied
    extra = supplied - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unexpected: {', '.join(sorted(extra))}")
        raise ValueError(f"Input features do not match the trained model ({'; '.join(details)}).")

    prepared: dict[str, int | float] = {}
    for feature in feature_columns:
        value = input_values[feature]
        if value is None:
            raise ValueError(f"{feature} is required.")
        if feature == "age":
            if value not in AGE_MAPPING:
                raise ValueError(f"Unsupported age group: {value}")
            prepared[feature] = AGE_MAPPING[value]
        elif feature == "gender":
            if value not in GENDER_MAPPING:
                raise ValueError(f"Unsupported gender: {value}")
            prepared[feature] = GENDER_MAPPING[value]
        elif feature in LABEL_ENCODED_FEATURES:
            prepared[feature] = encode_category(value, encoders[feature], feature)
        elif feature in MEDICATION_FEATURES:
            if value not in MEDICATION_MAPPING:
                raise ValueError(f"Unsupported medication status for {feature}: {value}")
            prepared[feature] = MEDICATION_MAPPING[value]
        elif feature == "change":
            if value not in CHANGE_MAPPING:
                raise ValueError(f"Unsupported change value: {value}")
            prepared[feature] = CHANGE_MAPPING[value]
        elif feature == "diabetesMed":
            if value not in BINARY_MAPPING:
                raise ValueError(f"Unsupported diabetes medication value: {value}")
            prepared[feature] = BINARY_MAPPING[value]
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{feature} must be numeric.")
            if value < 0:
                raise ValueError(f"{feature} cannot be negative.")
            prepared[feature] = value

    return pd.DataFrame([prepared], columns=feature_columns)
