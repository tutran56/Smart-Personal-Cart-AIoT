from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


FEATURE_COLUMNS = [
    "value", "hour", "dayofweek", "rolling_mean_12", "rolling_std_12",
    "rolling_mean_36", "delta_1", "delta_3", "zscore_rolling", "is_stuck_candidate"
]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load telemetry dataset. Expected columns: timestamp, value, optional label."""
    if path is None:
        candidates = [
            DATA_DIR / "ambient_temperature_system_failure_labeled.csv",
            DATA_DIR / "sample_ambient_temperature_system_failure.csv",
        ]
        for p in candidates:
            if p.exists():
                path = p
                break
        else:
            raise FileNotFoundError("Không tìm thấy dataset. Hãy chạy: python src/download_data.py")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    if "label" not in df.columns:
        df["label"] = 0
    df["label"] = df["label"].fillna(0).astype(int)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-series features for anomaly detection.

    Đây không phải phần data cleaning của Lab 2. Ở Lab 3, feature được dùng để
    mô tả hành vi gần của hệ thống: giá trị đang tăng/giảm nhanh, lệch khỏi
    pattern gần đây, hoặc cảm biến có dấu hiệu bị đứng.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["hour"] = out["timestamp"].dt.hour
    out["dayofweek"] = out["timestamp"].dt.dayofweek

    out["rolling_mean_12"] = out["value"].rolling(window=12, min_periods=1).mean()
    out["rolling_std_12"] = out["value"].rolling(window=12, min_periods=2).std().fillna(0)
    out["rolling_mean_36"] = out["value"].rolling(window=36, min_periods=1).mean()

    out["delta_1"] = out["value"].diff().fillna(0)
    out["delta_3"] = out["value"].diff(3).fillna(0)

    rolling_std_36 = out["value"].rolling(window=36, min_periods=2).std().replace(0, np.nan)
    out["zscore_rolling"] = ((out["value"] - out["rolling_mean_36"]) / rolling_std_36).fillna(0)

    # Nếu độ lệch trong nhiều điểm gần nhất gần bằng 0, cảm biến có thể bị stuck.
    out["is_stuck_candidate"] = (
        out["value"].rolling(window=6, min_periods=6).std().fillna(1) < 0.03
    ).astype(int)

    return out


def time_split(df: pd.DataFrame, train_ratio: float = 0.65) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically. Do not random split IoT time-series."""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def normalize_scores(raw_scores, score_min: float, score_max: float) -> np.ndarray:
    """Normalize raw anomaly scores into [0, 1] using training score distribution."""
    raw_scores = np.asarray(raw_scores, dtype=float)
    norm = (raw_scores - score_min) / (score_max - score_min + 1e-9)
    return np.clip(norm, 0.0, 1.0)


def event_type_from_row(row: pd.Series) -> str:
    """Map feature evidence to an event type that is easier for dashboard/operator."""
    if int(row.get("is_stuck_candidate", 0)) == 1:
        return "SENSOR_STUCK_CANDIDATE"
    if abs(float(row.get("delta_1", 0))) > 3:
        return "SUDDEN_JUMP_OR_DROP"
    if abs(float(row.get("zscore_rolling", 0))) > 3:
        return "TEMPERATURE_PATTERN_DEVIATION"
    return "TEMPERATURE_ANOMALY"


def severity_from_score(score: float, threshold: float, high_threshold: float = 0.80) -> str:
    """Convert anomaly score to operational severity."""
    if score >= high_threshold:
        return "HIGH"
    if score >= threshold:
        return "MEDIUM"
    return "LOW"


def decision_from_severity(severity: str) -> str:
    if severity == "HIGH":
        return "CREATE_ALERT_AND_REQUIRE_HUMAN_CHECK"
    if severity == "MEDIUM":
        return "CREATE_WARNING_EVENT"
    return "LOG_FOR_MONITORING"


def explanation_from_row(row: pd.Series) -> str:
    reason_parts = []
    if abs(float(row.get("zscore_rolling", 0))) > 3:
        reason_parts.append("value deviates strongly from recent pattern")
    if abs(float(row.get("delta_1", 0))) > 3:
        reason_parts.append("sudden jump/drop compared with previous reading")
    if int(row.get("is_stuck_candidate", 0)) == 1:
        reason_parts.append("sensor may be stuck")
    if not reason_parts:
        reason_parts.append("anomaly score is above operating threshold")
    return "; ".join(reason_parts)


def build_events(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Convert model outputs into AIoT anomaly events and decisions."""
    if threshold is None:
        threshold = 0.55

    events = []
    for _, row in df.iterrows():
        if int(row.get("is_anomaly", 0)) == 0:
            continue

        score = float(row.get("anomaly_score", 0.0))
        severity = severity_from_score(score, threshold=threshold)
        events.append({
            "timestamp": row["timestamp"],
            "device_id": row.get("device_id", "nab_office_temp_sensor_01"),
            "value": float(row["value"]),
            "anomaly_score": round(score, 4),
            "threshold_used": round(float(threshold), 4),
            "severity": severity,
            "event_type": event_type_from_row(row),
            "decision": decision_from_severity(severity),
            "explanation": explanation_from_row(row),
            "model_version": row.get("model_version", "iforest_v2")
        })

    return pd.DataFrame(events)


def evaluate_detection(y_true, y_pred) -> dict:
    """Evaluate anomaly detector if labels are available."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1_score": float(round(f1, 4)),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1])
    }


def make_windows(values: np.ndarray, window_size: int = 24) -> np.ndarray:
    """Create sliding windows for neural-network autoencoder demo."""
    values = np.asarray(values, dtype=float)
    if len(values) < window_size:
        raise ValueError("Không đủ dữ liệu để tạo window.")
    return np.array([values[i:i + window_size] for i in range(len(values) - window_size + 1)])


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
