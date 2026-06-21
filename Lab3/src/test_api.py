from __future__ import annotations

import json
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

DATA_FILE = ROOT / "data" / "ambient_temperature_system_failure_labeled.csv"
if not DATA_FILE.exists():
    DATA_FILE = ROOT / "data" / "sample_ambient_temperature_system_failure.csv"

df = pd.read_csv(DATA_FILE).tail(40)
history = [
    {"timestamp": str(row["timestamp"]), "value": float(row["value"]), "device_id": "nab_office_temp_sensor_01"}
    for _, row in df.iterrows()
]

print("Kiểm tra /health")
health = requests.get("http://127.0.0.1:8000/health", timeout=10).json()
print(health)

print("\nKiểm tra /model-info")
model_info = requests.get("http://127.0.0.1:8000/model-info", timeout=10).json()
print(model_info)

print("\nKiểm tra /detect-anomaly")
resp = requests.post("http://127.0.0.1:8000/detect-anomaly", json={"history": history}, timeout=10)
data = resp.json()
print(data)

assert "model_output" in data
assert "event" in data
assert "anomaly_score" in data["model_output"]
assert "threshold_used" in data["model_output"]
assert "decision" in data["event"]

(OUT / "api_test_result.json").write_text(
    json.dumps({"health": health, "model_info": model_info, "detect_anomaly": data}, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("\nPASS: API response schema hợp lệ.")
