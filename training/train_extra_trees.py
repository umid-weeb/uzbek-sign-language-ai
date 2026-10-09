from __future__ import annotations

import csv
import pickle
from pathlib import Path

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "gesture_extra_trees.pkl"
METRICS_PATH = ROOT / "data" / "processed" / "metrics_extra_trees.txt"
SEED = 42


def load_csv(path: Path) -> tuple[list[list[float]], list[str], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        if not columns or "label" not in columns:
            raise ValueError(f"{path} faylida label ustuni yo'q.")
        features = [column for column in columns if column != "label"]
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} bo'sh.")
    return (
        [[float(row[column]) for column in features] for row in rows],
        [row["label"] for row in rows],
        features,
    )


def main() -> None:
    x_train, y_train, feature_columns = load_csv(TRAIN_PATH)
    x_test, y_test, test_columns = load_csv(TEST_PATH)
    if feature_columns != test_columns:
        raise ValueError("Train/test feature ustunlari mos emas.")
    if set(y_test) - set(y_train):
        raise ValueError("Testda train tarkibida bo'lmagan klass bor.")

    x_train, y_train = shuffle(x_train, y_train, random_state=SEED)
    model = ExtraTreesClassifier(
        n_estimators=700,
        max_features=1.0,
        min_samples_leaf=1,
        bootstrap=False,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    print("ExtraTrees training boshlandi...")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "labels": list(model.classes_),
                "random_state": SEED,
                "keypoint_layout": "rtmlib_mmpose_wholebody_v1",
                "feature_mode": "wrist_relative",
            },
            file,
        )
    METRICS_PATH.write_text(
        f"Accuracy: {accuracy:.4%}\n"
        f"Train samples: {len(x_train)}\nTest samples: {len(x_test)}\n"
        f"Trees: {model.n_estimators}\nFeatures: {len(feature_columns)}\n\n"
        f"{report}",
        encoding="utf-8",
    )
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
