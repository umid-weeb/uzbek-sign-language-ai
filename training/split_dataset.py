
import csv
import random
from collections import Counter
from pathlib import Path

SEED = 42
TEST_RATIO = 0.20

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "dataset.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"


def main():
    random.seed(SEED)

    with DATASET_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "label" not in fieldnames:
        raise ValueError("CSV faylida label ustuni topilmadi.")

    if not rows:
        raise ValueError("Dataset bo'sh.")

    if any(not row.get("label") for row in rows):
        raise ValueError("Datasetda bo'sh label mavjud.")

    # Har bir klassni alohida ajratamiz.
    groups = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)

    train_rows = []
    test_rows = []

    # Har bir klass uchun 80/20 nisbatni saqlaymiz.
    for label in sorted(groups):
        samples = groups[label].copy()
        random.shuffle(samples)

        if len(samples) < 2:
            raise ValueError(
                f"{label} klassida split uchun namuna yetarli emas."
            )

        test_count = max(1, round(len(samples) * TEST_RATIO))
        test_count = min(test_count, len(samples) - 1)

        test_rows.extend(samples[:test_count])
        train_rows.extend(samples[test_count:])

    # Natijalarni aralashtiramiz.
    random.shuffle(train_rows)
    random.shuffle(test_rows)

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    for path, data in (
        (TRAIN_PATH, train_rows),
        (TEST_PATH, test_rows),
    ):
        with path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    print(f"Jami:  {len(rows)}")
    print(f"Train: {len(train_rows)}")
    print(f"Test:  {len(test_rows)}")
    print(f"Klasslar: {len(groups)}")
    print("\nTrain klasslari:", dict(Counter(
        row["label"] for row in train_rows
    )))
    print("\nTest klasslari:", dict(Counter(
        row["label"] for row in test_rows
    )))
    print("\nFayllar saqlandi:")
    print(TRAIN_PATH)
    print(TEST_PATH)


if __name__ == "__main__":
    main()
