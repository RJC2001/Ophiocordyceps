from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from core.config import MODELS_DIR, TEMP_DIR, MODEL_TEMP_DIR
from modules.model_manager import get_model_path


def load_model(model_name: str = "", upload_token: str = "", session_id: str = "") -> object:
    if model_name:
        model_path = get_model_path(model_name, "csv")
    elif upload_token:
        if not session_id:
            raise ValueError("session_id required with upload_token")
        model_path = MODEL_TEMP_DIR / f"{session_id}_test_{upload_token}.model"
        if not model_path.exists():
            raise FileNotFoundError(f"Uploaded model not found: {upload_token}")
    else:
        raise ValueError("Either model_name or upload_token required")
    return joblib.load(model_path)


def predict_csv(
    model: object,
    csv_path: str,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    csv_cols = set(df.columns)

    model_order = getattr(model, "feature_names_in_", None)

    if feature_cols:
        want = [c for c in feature_cols if c in csv_cols]
    elif model_order is not None:
        want = [c for c in model_order if c in csv_cols]
    else:
        want = list(csv_cols)

    if not want:
        raise ValueError("No usable feature columns found in CSV")

    if model_order is not None and len(want) == len(model_order) and all(c in model_order for c in want):
        X = df[list(model_order)]
    else:
        X = df[want]

    predictions = model.predict(X)
    result: dict[str, Any] = {
        "rows": df.to_dict(orient="records"),
        "predictions": predictions.tolist(),
    }

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            result["probabilities"] = proba.tolist()
    except Exception:
        pass

    return result
