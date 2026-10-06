from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal
from app.schemas.prediction import PredictionInput, PredictionRecordResponse, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/api", tags=["predictions"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/health")
def health_check():
    return {"status": "ok", "message": "Hospital readmission prediction service is running."}


@router.post("/predict", response_model=PredictionResponse)
def predict(input_payload: PredictionInput, db: Session = Depends(get_db)):
    service = PredictionService(db)
    payload = input_payload.model_dump(exclude_none=True)
    try:
        result = service.create_prediction(patient_reference=input_payload.patient_reference, payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}") from exc

    return PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        risk_level=result["risk_level"],
        patient_reference=result["patient_reference"],
        model_features=result["model_features"],
        created_at=result["created_at"],
    )


@router.get("/predictions", response_model=list[PredictionRecordResponse])
def list_predictions(limit: int = 10, db: Session = Depends(get_db)):
    service = PredictionService(db)
    rows = service.get_recent_predictions(limit=limit)
    return [
        PredictionRecordResponse(
            id=row["id"],
            patient_reference=row["patient_reference"],
            prediction=row["prediction"],
            probability=row["probability"],
            risk_level=row["risk_level"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
