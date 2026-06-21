from __future__ import annotations

"""Test API logic without starting uvicorn.

Use this when the classroom machine blocks local ports or when students want to verify
that app.py can load the model and return the correct response schema.
"""

import json
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

from app import app

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

DATA_FILE = ROOT / "data" / "ambient_temperature_system_failure_labeled.csv"
if not DATA_FILE.exists():
    DATA_FILE = ROOT / "data" / "sample_ambient_temperature_system_failure.csv"

df = pd.read_csv(DATA_FILE).tail(40)
history = [
    {
        "timestamp": str(row["timestamp"]),
        "value": float(row["value"]),
        "device_id": "nab_office_temp_sensor_01"
    }
    for _, row in df.iterrows()
]

client = TestClient(app)

print("Kiểm tra /health")
health = client.get("/health").json()
print(health)

print("\nKiểm tra /model-info")
model_info = client.get("/model-info").json()
print(model_info)

print("\nKiểm tra /detect-anomaly")
resp = client.post("/detect-anomaly", json={"history": history})
print(resp.status_code)
data = resp.json()
print(data)

assert resp.status_code == 200
assert "model_output" in data
assert "event" in data
assert "anomaly_score" in data["model_output"]
assert "threshold_used" in data["model_output"]
assert "decision" in data["event"]
assert "severity" in data["event"]

(OUT / "api_test_result.json").write_text(
    json.dumps({"health": health, "model_info": model_info, "detect_anomaly": data}, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("\nPASS: API response schema hợp lệ.")
print("Đã lưu outputs/api_test_result.json")
