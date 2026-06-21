from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv"
LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"
LOCAL_SAMPLE = DATA_DIR / "sample_ambient_temperature_system_failure.csv"
OUT_FILE = DATA_DIR / "ambient_temperature_system_failure_labeled.csv"


def _download_text(url: str, timeout: int = 20) -> str:
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def main() -> None:
    print("Đang tải dataset public NAB từ GitHub...")
    try:
        csv_text = _download_text(RAW_DATA_URL)
        raw_path = DATA_DIR / "ambient_temperature_system_failure.csv"
        raw_path.write_text(csv_text, encoding="utf-8")
        df = pd.read_csv(raw_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["label"] = 0

        try:
            labels_text = _download_text(LABELS_URL)
            labels = json.loads(labels_text)
            windows = labels.get("realKnownCause/ambient_temperature_system_failure.csv", [])
            for start, end in windows:
                start_ts = pd.to_datetime(start)
                end_ts = pd.to_datetime(end)
                df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts), "label"] = 1
            print(f"Đã gắn nhãn anomaly từ NAB combined_windows.json: {len(windows)} window.")
        except Exception as label_error:
            print("Không tải được file label. Vẫn lưu dataset, label mặc định = 0.")
            print("Chi tiết:", label_error)

        df.to_csv(OUT_FILE, index=False)
        print(f"Đã lưu dữ liệu public tại: {OUT_FILE}")
        print(df.head())
    except Exception as e:
        print("Không thể tải dataset public trong môi trường hiện tại.")
        print("Sử dụng file sample kèm theo project để vẫn chạy được bài mẫu.")
        print("Chi tiết:", e)
        if not LOCAL_SAMPLE.exists():
            raise FileNotFoundError("Thiếu file sample fallback.")
        df = pd.read_csv(LOCAL_SAMPLE)
        df.to_csv(OUT_FILE, index=False)
        print(f"Đã copy sample dataset sang: {OUT_FILE}")


if __name__ == "__main__":
    main()
