from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


def make_history_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "Timestamp (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Prediction": "Readmitted" if result["prediction"] == 1 else "Not readmitted",
        "Probability": result["probability"],
        "Risk level": result["risk_level"],
    }


def history_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        records,
        columns=["Timestamp (UTC)", "Prediction", "Probability", "Risk level"],
    )
