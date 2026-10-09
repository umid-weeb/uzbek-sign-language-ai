from __future__ import annotations

import csv
import pickle
from pathlib import Path

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN = ROOT / "data" / "processed" / "train.csv"
CALIBRATION = ROOT / "data" / "camera_calibration.csv"
TEST = ROOT / "data" / "processed" / "test.csv"
MODEL = ROOT / "models" / "gesture_extra_trees_camera.pkl"
METRICS = ROOT / "data" / "processed" / "metrics_extra_trees_camera.txt"


def read(path: Path) -> tuple[list[list[float]], list[str], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows or "label" not in rows[0]:
        raise ValueError(f"Dataset noto'g'ri yoki bo'sh: {path}")
    cols = [c for c in rows[0] if c != "label"]
    values = [[float(row[c]) for c in cols] for row in rows]
    if any(not all(map(lambda value: value == value and abs(value) != float("inf"), row)) for row in values):
        raise ValueError(f"NaN yoki infinity topildi: {path}")
    return values, [row["label"] for row in rows], cols


def main() -> None:
    if not CALIBRATION.exists():
        raise FileNotFoundError("Avval pose/camera_calibration.py bilan calibration yig'ing.")
    x_base, y_base, cols = read(BASE_TRAIN)
    x_cal, y_cal, cal_cols = read(CALIBRATION)
    x_test, y_test, test_cols = read(TEST)
    if cols != cal_cols or cols != test_cols:
        raise ValueError("Calibration/train/test feature ustunlari mos emas.")
    expected_labels = set(y_base)
    if set(y_cal) != expected_labels or set(y_test) != expected_labels:
        raise ValueError("Train, calibration va test label to'plamlari bir xil bo'lishi kerak.")
    if len(x_cal) < len(expected_labels) * 20:
        raise ValueError("Har bir Uzbek harfi uchun kamida 20 ta calibration sample kerak.")
    x_train, y_train = shuffle(x_base + x_cal, y_base + y_cal, random_state=42)
    model = ExtraTreesClassifier(
        n_estimators=700, max_features=1.0, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    with MODEL.open("wb") as file:
        pickle.dump({"model": model, "feature_columns": cols, "labels": list(model.classes_),
                     "keypoint_layout": "rtmlib_mmpose_wholebody_v1",
                     "feature_mode": "wrist_relative",
                     "calibration_rows": len(x_cal)}, file)
    METRICS.write_text(
        f"Base train: {len(x_base)}\nCalibration train: {len(x_cal)}\n"
        f"Test: {len(x_test)}\nAccuracy: {accuracy:.4%}\n\n"
        + classification_report(y_test, prediction, zero_division=0),
        encoding="utf-8",
    )
    print(f"Camera-calibrated model accuracy on held-out test: {accuracy:.2%}")
    print(f"Saved: {MODEL}")


if __name__ == "__main__":
    main()
