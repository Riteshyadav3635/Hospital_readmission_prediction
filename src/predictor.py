from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.preprocessing import prepare_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "ml" / "models"


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    """Load and validate the original fitted artifacts once per process."""
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    pca = joblib.load(MODEL_DIR / "pca_95.pkl")
    model = joblib.load(MODEL_DIR / "hospital_readmission_model.pkl")
    encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")

    feature_columns = list(getattr(scaler, "feature_names_in_", []))
    if not feature_columns:
        raise ValueError("The saved scaler does not include its fitted feature names.")
    if getattr(pca, "n_features_in_", None) != len(feature_columns):
        raise ValueError("The saved PCA input size does not match the scaler features.")
    if getattr(model, "n_features_in_", None) != getattr(pca, "n_components_", None):
        raise ValueError("The saved classifier input size does not match the PCA output.")
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("The saved classifier must support predict and predict_proba.")

    for key in ("race", "medical_specialty", "diag_1", "diag_2", "diag_3"):
        if key not in encoders:
            raise ValueError(f"The saved label encoders are missing {key}.")

    return {
        "scaler": scaler,
        "pca": pca,
        "model": model,
        "encoders": encoders,
        "feature_columns": feature_columns,
    }


def predict(input_values: dict[str, Any]) -> dict[str, Any]:
    """Run existing feature mappings, scaling, PCA, and Logistic Regression."""
    bundle = load_model_bundle()
    frame = prepare_features(
        input_values,
        bundle["feature_columns"],
        bundle["encoders"],
    )
    scaled = bundle["scaler"].transform(frame)
    components = bundle["pca"].transform(scaled)
    model = bundle["model"]

    predicted_class = int(model.predict(components)[0])
    probability: float | None = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(model.predict_proba(components)[0, classes.index(1)])

    positive_prediction = predicted_class == 1
    return {
        "prediction": predicted_class,
        "probability": probability,
        "risk_level": "Higher estimated readmission risk" if positive_prediction else "Lower estimated readmission risk",
        "explanation": (
            "The existing model classified this encounter as readmitted within 30 days."
            if positive_prediction
            else "The existing model did not classify this encounter as readmitted within 30 days."
        ),
        "feature_frame": frame,
    }
