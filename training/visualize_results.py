
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "gesture_classifier.pkl"
OUTPUT_DIR = ROOT / "data" / "processed" / "visualizations"


def load_test_data():
    with TEST_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Test dataset bo'sh.")

    with MODEL_PATH.open("rb") as file:
        saved = pickle.load(file)

    model = saved["model"]
    feature_columns = saved["feature_columns"]

    X = [
        [float(row[column]) for column in feature_columns]
        for row in rows
    ]
    y_true = [row["label"] for row in rows]

    return model, X, y_true


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, X_test, y_true = load_test_data()

    print(f"Test boshlandi: {len(y_true)} ta namuna")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_true, y_pred)

    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    correct = sum(a == b for a, b in zip(y_true, y_pred))
    incorrect = len(y_true) - correct

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"To'g'ri: {correct}")
    print(f"Xato: {incorrect}")

    # 1. Umumiy natijalar grafigi
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Correct", "Incorrect"],
        [correct, incorrect],
    )
    ax.set_title(f"Test Results — Accuracy: {accuracy:.2%}")
    ax.set_ylabel("Samples")

    for bar, value in zip(bars, [correct, incorrect]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value} ({value / len(y_true):.1%})",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_results.png", dpi=160)
    plt.close(fig)

    # 2. Har bir klassning F1-score grafigi
    f1_scores = [
        report[label]["f1-score"] for label in labels
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(labels, np.array(f1_scores) * 100)
    ax.set_title("F1-score by Gesture Class")
    ax.set_ylabel("F1-score (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "f1_scores.png", dpi=160)
    plt.close(fig)

    # 3. Confusion matrix
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(14, 12))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )
    display.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
        values_format="d",
    )
    ax.set_title("Confusion Matrix — Test Dataset")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=160,
    )
    plt.close(fig)

    print("\nGrafiklar saqlandi:")
    print(OUTPUT_DIR / "test_results.png")
    print(OUTPUT_DIR / "f1_scores.png")
    print(OUTPUT_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
