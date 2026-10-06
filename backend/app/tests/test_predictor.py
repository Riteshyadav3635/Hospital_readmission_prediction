from app.ml.predictor import get_model_bundle, predict_payload


def test_predict_payload_returns_valid_risk_score():
    result = predict_payload(
        {
            "race": "Caucasian",
            "gender": "Female",
            "age_group": "[50-60)",
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
            "metformin": "No",
            "repaglinide": "No",
            "nateglinide": "No",
            "chlorpropamide": "No",
            "glimepiride": "No",
            "acetohexamide": "No",
            "glipizide": "No",
            "glyburide": "No",
            "tolbutamide": "No",
            "pioglitazone": "No",
            "rosiglitazone": "No",
            "acarbose": "No",
            "miglitol": "No",
            "troglitazone": "No",
            "tolazamide": "No",
            "insulin": "No",
            "glyburide_metformin": "No",
            "glipizide_metformin": "No",
            "glimepiride_pioglitazone": "No",
            "metformin_rosiglitazone": "No",
            "metformin_pioglitazone": "No",
            "change": "No",
            "diabetesMed": "Yes",
        }
    )

    assert "prediction" in result
    assert "probability" in result
    assert "risk_level" in result
    assert 0 <= result["probability"] <= 1
    assert result["prediction"] in (0, 1)
    assert result["model_features"]["age"] == 5


def test_model_bundle_loads_once():
    first = get_model_bundle()
    second = get_model_bundle()
    assert first["model"] is second["model"]
