from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "external"
    / "mrkomiljon-uzbek-sign-language"
    / "data.pickle"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "external"
    / "mrkomiljon-uzbek-sign-language"
    / "dataset.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the downloaded public Uzbek landmark pickle to CSV."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.source.open("rb") as file:
        payload = pickle.load(file)
    data = payload.get("data")
    labels = payload.get("labels")
    if not isinstance(data, list) or not isinstance(labels, list):
        raise ValueError("Public dataset data/labels list formatida emas.")
    if len(data) != len(labels) or not data:
        raise ValueError("Public dataset data va labels uzunligi mos emas.")

    feature_count = len(data[0])
    if feature_count != 42 or any(len(row) != feature_count for row in data):
        raise ValueError("Public dataset 42 ta bir xil feature'dan iborat bo'lishi kerak.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_label"] + [
        f"x{i}" if i % 2 == 0 else f"y{i // 2}"
        for i in range(feature_count)
    ]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        for label, row in zip(labels, data):
            writer.writerow([label, *row])

    print(f"Converted {len(data)} samples to {args.output}")


if __name__ == "__main__":
    main()
