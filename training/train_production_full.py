from __future__ import annotations

import csv
import pickle
from pathlib import Path

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset.csv"
MODEL = ROOT / "models" / "gesture_extra_trees_full.pkl"
SEED = 42


def load_dataset(path: Path) -> tuple[list[list[float]], list[str], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows or "label" not in rows[0]:
        raise ValueError(f"Dataset noto'g'ri yoki bo'sh: {path}")
    columns = [column for column in rows[0] if column != "label"]
    values = [[float(row[column]) for column in columns] for row in rows]
    labels = [row["label"] for row in rows]
    if len(columns) != 42 or any(len(row) != 42 for row in values):
        raise ValueError("Production dataset 42 ta numeric feature bo'lishi kerak.")
    return values, labels, columns


def main() -> None:
    features, labels, columns = load_dataset(DATASET)
    features, labels = shuffle(features, labels, random_state=SEED)
    model = ExtraTreesClassifier(
        n_estimators=1000,
        max_features=1.0,
        min_samples_leaf=1,
        bootstrap=False,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    print(f"Production training boshlandi: {len(features)} sample, {len(set(labels))} klass")
    model.fit(features, labels)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    with MODEL.open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "feature_columns": columns,
                "labels": list(model.classes_),
                "random_state": SEED,
                "keypoint_layout": "rtmlib_mmpose_wholebody_v1",
                "feature_mode": "wrist_relative",
                "training_scope": "all_labeled_uzbek_dataset",
                "training_samples": len(features),
            },
            file,
        )
    print(f"Production model saqlandi: {MODEL}")


if __name__ == "__main__":
    main()
