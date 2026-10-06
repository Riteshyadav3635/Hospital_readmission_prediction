from __future__ import annotations

from typing import Any

import streamlit as st

from src.predictor import load_model_bundle, predict
from src.preprocessing import (
    AGE_MAPPING,
    BINARY_MAPPING,
    CHANGE_MAPPING,
    GENDER_MAPPING,
    LABEL_ENCODED_FEATURES,
    MEDICATION_FEATURES,
    MEDICATION_MAPPING,
    default_inputs,
)
from src.utils import history_frame, make_history_record

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide",
)

DISCLAIMER = (
    "This application is intended for educational and research purposes only and "
    "should not be used as a substitute for professional medical advice."
)

NUMERIC_LABELS = {
    "admission_type_id": "Admission type ID",
    "discharge_disposition_id": "Discharge disposition ID",
    "admission_source_id": "Admission source ID",
    "time_in_hospital": "Days in hospital",
    "num_lab_procedures": "Lab procedures",
    "num_procedures": "Procedures",
    "num_medications": "Medications",
    "number_outpatient": "Outpatient visits",
    "number_emergency": "Emergency visits",
    "number_inpatient": "Inpatient visits",
    "number_diagnoses": "Number of diagnoses",
}


def _display_label(feature: str) -> str:
    if feature in NUMERIC_LABELS:
        return NUMERIC_LABELS[feature]
    return feature.replace("-", " ").replace("_", " ").replace("Id", "ID").title()


def _numeric_input(feature: str, defaults: dict[str, int]) -> int:
    return int(
        st.number_input(
            _display_label(feature),
            min_value=0,
            value=max(0, int(defaults.get(feature, 0))),
            step=1,
            key=f"feature_{feature}",
            help=f"Model input feature: {feature}",
        )
    )


def _category_input(
    feature: str,
    options: list[str],
    default: str | None = None,
) -> str:
    if not options:
        raise ValueError(f"No trained categories are available for {feature}.")
    index = options.index(default) if default in options else 0
    return st.selectbox(
        _display_label(feature),
        options=options,
        index=index,
        key=f"feature_{feature}",
        help=f"Model input feature: {feature}",
    )


def collect_model_inputs(bundle: dict[str, Any]) -> dict[str, Any]:
    """Render controls for every actual feature expected by the saved scaler."""
    feature_columns = bundle["feature_columns"]
    encoders = bundle["encoders"]
    defaults = default_inputs(encoders)
    values: dict[str, Any] = {}

    st.caption(
        "Enter values for the 41 features expected by the saved model. "
        "Feature labels and category options come from the training artifacts."
    )

    tabs = st.tabs(["Encounter", "Diagnoses & medications", "All model fields"])
    with tabs[0]:
        columns = st.columns(3)
        encounter_features = [
            feature
            for feature in feature_columns
            if feature not in LABEL_ENCODED_FEATURES
            and feature not in MEDICATION_FEATURES
            and feature not in {"age", "gender", "change", "diabetesMed"}
        ]
        with columns[0]:
            values["race"] = _category_input(
                "race",
                [str(value) for value in encoders["race"].classes_],
                defaults["race"],
            )
            values["gender"] = _category_input(
                "gender", list(GENDER_MAPPING), defaults["gender"]
            )
            values["age"] = _category_input(
                "age", list(AGE_MAPPING), defaults["age"]
            )
        for index, feature in enumerate(encounter_features):
            column = columns[(index + 1) % len(columns)]
            with column:
                values[feature] = _numeric_input(feature, defaults)

    with tabs[1]:
        columns = st.columns(3)
        category_features = [
            feature
            for feature in feature_columns
            if feature in LABEL_ENCODED_FEATURES
            and feature != "race"
        ]
        category_features += [
            feature
            for feature in feature_columns
            if feature in MEDICATION_FEATURES or feature in {"change", "diabetesMed"}
        ]
        for index, feature in enumerate(category_features):
            with columns[index % len(columns)]:
                if feature in LABEL_ENCODED_FEATURES:
                    options = [str(value) for value in encoders[feature].classes_]
                    values[feature] = _category_input(
                        feature, options, defaults.get(feature)
                    )
                elif feature in MEDICATION_FEATURES:
                    values[feature] = _category_input(
                        feature, list(MEDICATION_MAPPING), defaults[feature]
                    )
                elif feature == "change":
                    values[feature] = _category_input(
                        feature, list(CHANGE_MAPPING), defaults[feature]
                    )
                else:
                    values[feature] = _category_input(
                        feature, list(BINARY_MAPPING), defaults[feature]
                    )

    with tabs[2]:
        st.write(
            "Review all model fields and adjust any values above. "
            "The full feature table is kept in the saved scaler's feature order."
        )
        st.dataframe(
            {
                "Feature": feature_columns,
                "Type": [
                    "Categorical" if column in {
                        *LABEL_ENCODED_FEATURES,
                        *MEDICATION_FEATURES,
                        "age",
                        "gender",
                        "change",
                        "diabetesMed",
                    } else "Numeric"
                    for column in feature_columns
                ],
            },
            hide_index=True,
            width="stretch",
        )

    return values


def render_home(bundle: dict[str, Any]) -> None:
    st.title("Hospital Readmission Prediction")
    st.subheader("A research demonstration using the project's saved ML pipeline")
    st.write(
        "This application estimates whether a hospital encounter may be followed "
        "by readmission within 30 days, using the existing trained model artifacts."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("### Purpose")
        st.write(
            "Explore an existing machine-learning workflow for hospital readmission "
            "research. The output is a model estimate, not a diagnosis or a care recommendation."
        )
        st.markdown("### Machine-learning approach")
        st.write(
            "Saved categorical encoders and feature mappings feed the fitted "
            "StandardScaler, PCA transform, and Logistic Regression classifier."
        )
    with right:
        st.markdown("### Dataset")
        st.write(
            "The bundled CSV contains 101,766 inpatient encounter rows and a "
            "`readmitted` target column, alongside the original analysis notebooks. "
            "The app does not load the dataset at runtime or infer new performance metrics."
        )
        st.markdown("### Loaded artifact contract")
        st.write(
            f"{len(bundle['feature_columns'])} ordered scaler input features → "
            f"{bundle['pca'].n_components_} PCA components → "
            f"{bundle['model'].__class__.__name__}."
        )

    st.warning(DISCLAIMER)


def render_prediction(bundle: dict[str, Any]) -> None:
    st.title("Make a prediction")
    st.warning(DISCLAIMER)
    with st.form("prediction_form"):
        input_values = collect_model_inputs(bundle)
        submitted = st.form_submit_button(
            "Run readmission estimate", type="primary", width="stretch"
        )

    if submitted:
        try:
            result = predict(input_values)
        except (ValueError, KeyError, TypeError) as error:
            st.error(f"Input could not be processed: {error}")
            return

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.insert(0, make_history_record(result))

        st.markdown("## Result")
        if result["prediction"] == 1:
            st.error("The model estimated a higher readmission risk.")
        else:
            st.success("The model estimated a lower readmission risk.")
        st.metric("Prediction", "Readmitted" if result["prediction"] else "Not readmitted")
        if result["probability"] is not None:
            st.metric(
                "Estimated probability of readmission",
                f"{result['probability']:.1%}",
            )
        st.caption(result["explanation"])
        st.info(DISCLAIMER)


def render_history() -> None:
    st.title("Prediction history")
    st.caption(
        "History contains prediction outputs only; patient input values are not saved. "
        "Streamlit Community Cloud storage is temporary and may be cleared when the app restarts."
    )
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Clear history", type="secondary"):
        st.session_state.history.clear()
        st.rerun()

    if st.session_state.history:
        st.dataframe(
            history_frame(st.session_state.history),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No predictions have been recorded in this session.")


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Home", "Prediction", "History"],
        label_visibility="collapsed",
    )
    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"The trained model artifacts could not be loaded: {error}")
        st.stop()

    if page == "Home":
        render_home(bundle)
    elif page == "Prediction":
        render_prediction(bundle)
    else:
        render_history()


if __name__ == "__main__":
    main()
