from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.database.repository import PredictionRepository
from app.ml.predictor import predict_payload


class PredictionService:
    def __init__(self, session: Session):
        self.session = session

    def create_prediction(self, *, patient_reference: str | None, payload: dict):
        result = predict_payload(payload)
        repository = PredictionRepository(self.session)
        record = repository.create(
            patient_reference=patient_reference,
            input_features=result["model_features"],
            prediction=result["prediction"],
            probability=result["probability"],
            risk_level=result["risk_level"],
        )
        return {
            "prediction": record.prediction,
            "probability": record.probability,
            "risk_level": record.risk_level,
            "patient_reference": record.patient_reference,
            "model_features": record.input_features,
            "created_at": record.created_at.astimezone(timezone.utc).isoformat(),
        }

    def get_recent_predictions(self, limit: int = 10):
        repository = PredictionRepository(self.session)
        rows = repository.list_recent(limit=limit)
        return [
            {
                "id": row.id,
                "patient_reference": row.patient_reference,
                "prediction": row.prediction,
                "probability": row.probability,
                "risk_level": row.risk_level,
                "created_at": row.created_at.astimezone(timezone.utc).isoformat(),
            }
            for row in rows
        ]
