from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from threading import RLock
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "prediction_history.csv"
HISTORY_COLUMNS = ["Timestamp (UTC)", "Prediction", "Probability", "Risk level"]
_HISTORY_LOCK = RLock()


def make_history_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "Timestamp (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Prediction": "Readmitted" if result["prediction"] == 1 else "Not readmitted",
        "Probability": result["probability"],
        "Risk level": result["risk_level"],
    }


def history_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=HISTORY_COLUMNS)


def load_prediction_history(path: Path = HISTORY_PATH) -> pd.DataFrame:
    """Read local prediction outcomes without caching mutable history."""
    if not path.exists():
        return history_frame([])
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return history_frame([])
    return frame.reindex(columns=HISTORY_COLUMNS)


def append_prediction_history(
    record: dict[str, Any], path: Path = HISTORY_PATH
) -> None:
    """Append an outcome atomically; input feature values are never persisted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _HISTORY_LOCK:
        existing = load_prediction_history(path)
        new_record = history_frame([record])
        updated = (
            new_record
            if existing.empty
            else pd.concat([existing, new_record], ignore_index=True)
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                suffix=".tmp",
                prefix="prediction_history_",
                dir=path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                updated.to_csv(temporary_file, index=False)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def clear_prediction_history(path: Path = HISTORY_PATH) -> None:
    with _HISTORY_LOCK:
        path.unlink(missing_ok=True)
