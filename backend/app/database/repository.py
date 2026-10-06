from typing import Any

from sqlalchemy.orm import Session

from app.database.models import PredictionRecord


class PredictionRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, *, patient_reference: str | None, input_features: dict[str, Any], prediction: int, probability: float, risk_level: str) -> PredictionRecord:
        record = PredictionRecord(
            patient_reference=patient_reference,
            input_features=input_features,
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def list_recent(self, limit: int = 10):
        return self.session.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit).all()
