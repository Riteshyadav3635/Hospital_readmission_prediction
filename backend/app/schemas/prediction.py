from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    patient_reference: str | None = Field(default=None, description="Optional patient identifier or encounter reference.")
    race: str = "Caucasian"
    gender: str = "Female"
    age_group: str = "[50-60)"
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    time_in_hospital: int = 4
    medical_specialty: str = "InternalMedicine"
    num_lab_procedures: int = 43
    num_procedures: int = 1
    num_medications: int = 16
    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = 0
    diag_1: str = "276"
    diag_2: str = "133"
    diag_3: str = "86"
    number_diagnoses: int = 8
    metformin: str = "No"
    repaglinide: str = "No"
    nateglinide: str = "No"
    chlorpropamide: str = "No"
    glimepiride: str = "No"
    acetohexamide: str = "No"
    glipizide: str = "No"
    glyburide: str = "No"
    tolbutamide: str = "No"
    pioglitazone: str = "No"
    rosiglitazone: str = "No"
    acarbose: str = "No"
    miglitol: str = "No"
    troglitazone: str = "No"
    tolazamide: str = "No"
    insulin: str = "No"
    glyburide_metformin: str = "No"
    glipizide_metformin: str = "No"
    glimepiride_pioglitazone: str = "No"
    metformin_rosiglitazone: str = "No"
    metformin_pioglitazone: str = "No"
    change: str = "No"
    diabetesMed: str = "Yes"


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    probability: float
    risk_level: str
    patient_reference: str | None = None
    model_features: dict[str, Any]
    created_at: str | None = None


class PredictionRecordResponse(BaseModel):
    id: int
    patient_reference: str | None
    prediction: int
    probability: float
    risk_level: str
    created_at: str
