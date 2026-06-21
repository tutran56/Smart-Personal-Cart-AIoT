from __future__ import annotations

import json
import time
from typing import List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .utils import (
    MODEL_DIR, OUTPUT_DIR, FEATURE_COLUMNS, add_time_features, normalize_scores,
    event_type_from_row, severity_from_score, decision_from_severity, explanation_from_row
)

MODEL_BUNDLE_PATH = MODEL_DIR / "anomaly_model_bundle_iforest_v2.joblib"
LEGACY_MODEL_PATH = MODEL_DIR / "isolation_forest_iforest_v1.joblib"
METRICS_PATH = OUTPUT_DIR / "iforest_metrics.json"

app = FastAPI(
    title="LAB 3 AIoT Anomaly Detection API",
    description="Demo deploy anomaly model: telemetry history -> anomaly_score -> event -> severity -> decision",
    version="2.0.0"
)

model_bundle = None
if MODEL_BUNDLE_PATH.exists():
    model_bundle = joblib.load(MODEL_BUNDLE_PATH)
elif LEGACY_MODEL_PATH.exists():
    # Fallback để không làm hỏng bài nếu sinh viên dùng package cũ.
    pipeline = joblib.load(LEGACY_MODEL_PATH)
    model_bundle = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "score_min": 0.0,
        "score_max": 1.0,
        "threshold": 0.55,
        "model_version": "iforest_legacy"
    }


class TelemetryPoint(BaseModel):
    timestamp: str = Field(..., examples=["2013-07-05 09:00:00"])
    value: float = Field(..., examples=[27.5])
    device_id: str = Field("nab_office_temp_sensor_01", examples=["room_temp_01"])


class AnomalyRequest(BaseModel):
    history: List[TelemetryPoint] = Field(
        ...,
        description="Danh sách telemetry gần nhất. Nên gửi tối thiểu 36 điểm để rolling feature ổn định."
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_bundle is not None,
        "model_bundle_path": str(MODEL_BUNDLE_PATH),
    }


@app.get("/model-info")
def model_info():
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    if model_bundle is None:
        return {
            "model_loaded": False,
            "message": "Chưa có model. Hãy chạy: python src/download_data.py && python src/train_anomaly.py"
        }

    return {
        "model_loaded": True,
        "model_name": "IsolationForest anomaly detector",
        "model_version": model_bundle.get("model_version", "unknown"),
        "input": "history of telemetry points with timestamp and value",
        "output": "anomaly_score, is_anomaly, threshold_used, event_type, severity, decision",
        "threshold": round(float(model_bundle.get("threshold", 0.55)), 4),
        "feature_columns": model_bundle.get("feature_columns", FEATURE_COLUMNS),
        "metrics": metrics
    }


@app.post("/detect-anomaly")
def detect_anomaly(payload: AnomalyRequest):
    if model_bundle is None:
        return {"error": "Model chưa được train. Hãy chạy: python src/train_anomaly.py"}

    start = time.time()
    pipeline = model_bundle["pipeline"]
    feature_columns = model_bundle.get("feature_columns", FEATURE_COLUMNS)
    threshold = float(model_bundle.get("threshold", 0.55))
    score_min = float(model_bundle.get("score_min", 0.0))
    score_max = float(model_bundle.get("score_max", 1.0))

    df = pd.DataFrame([p.model_dump() for p in payload.history])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_time_features(df)
    latest = df.iloc[[-1]].copy()

    raw_score = -pipeline.named_steps["detector"].score_samples(
        pipeline.named_steps["scaler"].transform(latest[feature_columns])
    )[0]
    score = float(normalize_scores([raw_score], score_min, score_max)[0])
    is_anomaly = score >= threshold

    latest_row = latest.iloc[-1]
    severity = severity_from_score(score, threshold=threshold)
    event_type = event_type_from_row(latest_row) if is_anomaly else "NORMAL_TELEMETRY"
    decision = decision_from_severity(severity) if is_anomaly else "NO_ALERT"
    explanation = explanation_from_row(latest_row) if is_anomaly else "telemetry is within learned normal operating pattern"

    warnings = []
    if len(payload.history) < 36:
        warnings.append("history có ít hơn 36 điểm; rolling feature có thể chưa ổn định.")

    return {
        "model_output": {
            "raw_score": round(float(raw_score), 6),
            "anomaly_score": round(score, 4),
            "threshold_used": round(threshold, 4),
            "is_anomaly": bool(is_anomaly),
            "model_version": model_bundle.get("model_version", "iforest_v2")
        },
        "event": {
            "event_type": event_type,
            "device_id": payload.history[-1].device_id,
            "timestamp": str(payload.history[-1].timestamp),
            "value": payload.history[-1].value,
            "severity": severity,
            "decision": decision,
            "explanation": explanation,
            "safety_note": "Không tự động điều khiển thiết bị khi anomaly cao; cần xác nhận hoặc rule an toàn."
        },
        "api_check": {
            "latency_ms": round((time.time() - start) * 1000, 2),
            "input_points": len(payload.history),
            "warnings": warnings
        }
    }
