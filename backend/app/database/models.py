from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
