from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "external" / "mrkomiljon-uzbek-sign-language"
DEFAULT_SOURCE = DATASET_DIR / "dataset.csv"
DEFAULT_TRAIN = DATASET_DIR / "train.csv"
DEFAULT_TEST = DATASET_DIR / "test.csv"
SEED = 42
TEST_RATIO = 0.20
LABEL_COLUMN = "source_label"


def _split_label_rows(rows: list[dict[str, str]], rng: random.Random) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split one label while keeping identical feature rows in one partition."""
    groups: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    feature_names = [name for name in rows[0] if name != LABEL_COLUMN]
    for row in rows:
        groups[tuple(row[name] for name in feature_names)].append(row)

    target_test = round(len(rows) * TEST_RATIO)
    group_rows = list(groups.values())
    rng.shuffle(group_rows)
    group_rows.sort(key=len, reverse=True)

    test_groups: list[list[dict[str, str]]] = []
    test_count = 0
    for group in group_rows:
        if abs((test_count + len(group)) - target_test) < abs(test_count - target_test):
            test_groups.append(group)
            test_count += len(group)

    test_ids = {id(row) for group in test_groups for row in group}
    test = [row for row in rows if id(row) in test_ids]
    train = [row for row in rows if id(row) not in test_ids]
    if not train or not test:
        raise ValueError("Har bir klassda train va test uchun namuna qolishi kerak.")
    return train, test


def split_dataset(source: Path, train_path: Path, test_path: Path) -> tuple[int, int]:
    with source.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or LABEL_COLUMN not in fieldnames:
        raise ValueError(f"CSV faylida {LABEL_COLUMN!r} ustuni topilmadi.")
    if not rows:
        raise ValueError("External dataset bo'sh.")
    if any(not row.get(LABEL_COLUMN) for row in rows):
        raise ValueError("External datasetda bo'sh source_label mavjud.")

    rng = random.Random(SEED)
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[LABEL_COLUMN]].append(row)

    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for label in sorted(groups, key=lambda value: int(value) if value.isdigit() else value):
        train, test = _split_label_rows(groups[label], rng)
        train_rows.extend(train)
        test_rows.extend(test)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    for path, output_rows in ((train_path, train_rows), (test_path, test_rows)):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    train_features = {(tuple(row[name] for name in fieldnames if name != LABEL_COLUMN)) for row in train_rows}
    test_features = {(tuple(row[name] for name in fieldnames if name != LABEL_COLUMN)) for row in test_rows}
    overlap = train_features & test_features
    if overlap:
        raise RuntimeError(f"Train/test orasida {len(overlap)} ta duplicate feature topildi.")

    print(f"Jami: {len(rows)} | Train: {len(train_rows)} | Test: {len(test_rows)}")
    print(f"Klasslar: {len(groups)} | Seed: {SEED} | Test ratio: {TEST_RATIO:.0%}")
    print("Train taqsimoti:", dict(sorted(Counter(row[LABEL_COLUMN] for row in train_rows).items(), key=lambda item: int(item[0]))))
    print("Test taqsimoti:", dict(sorted(Counter(row[LABEL_COLUMN] for row in test_rows).items(), key=lambda item: int(item[0]))))
    print(f"Train file: {train_path}")
    print(f"Test file:  {test_path}")
    return len(train_rows), len(test_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the downloaded Uzbek landmark dataset into leakage-safe train/test CSVs.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    args = parser.parse_args()
    split_dataset(args.source, args.train, args.test)


if __name__ == "__main__":
    main()
