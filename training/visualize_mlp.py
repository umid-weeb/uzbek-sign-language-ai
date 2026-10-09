
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data/processed/test.csv"
MODEL_PATH = ROOT / "archive" / "legacy-models" / "gesture_mlp.pt"
OUT_DIR = ROOT / "data/processed/visualizations"


class GestureMLP(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def load_test(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Test dataset bo‘sh.")

    columns = [c for c in rows[0] if c != "label"]
    X = np.array(
        [[float(row[c]) for c in columns] for row in rows],
        dtype=np.float32,
    )
    y = np.array([row["label"] for row in rows])
    return X, y, columns


def main():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    X, y_true, columns = load_test(TEST_PATH)

    if columns != checkpoint["feature_columns"]:
        raise ValueError("Test ustunlari model ustunlariga mos emas.")

    if X.shape[1] != checkpoint["input_size"]:
        raise ValueError("Feature soni model input_size qiymatiga mos emas.")

    labels = checkpoint["labels"]
    encoder = LabelEncoder()
    encoder.classes_ = np.array(labels)
    y_true_encoded = encoder.transform(y_true)

    mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.array(checkpoint["scaler_scale"], dtype=np.float32)
    X = (X - mean) / scale

    model = GestureMLP(
        checkpoint["input_size"],
        len(labels),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        y_pred = logits.argmax(dim=1).numpy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(
        y_true_encoded,
        y_pred,
        labels=np.arange(len(labels)),
    )

    fig, ax = plt.subplots(figsize=(16, 14))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    display.plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title("MLP Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mlp_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Per-class F1
    scores = f1_score(
        y_true_encoded,
        y_pred,
        labels=np.arange(len(labels)),
        average=None,
        zero_division=0,
    )

    order = np.argsort(scores)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(
        np.array(labels)[order],
        scores[order],
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score")
    ax.set_title("MLP F1 Score by Class")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mlp_f1_scores.png", dpi=160)
    plt.close(fig)

    # Classification report
    report = classification_report(
        y_true_encoded,
        y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        zero_division=0,
    )

    report_path = ROOT / "data/processed/metrics_mlp_visualization.txt"
    report_path.write_text(
        f"Accuracy: {np.mean(y_true_encoded == y_pred):.4%}\n\n"
        f"Classification report:\n{report}\n",
        encoding="utf-8",
    )

    print(f"Accuracy: {np.mean(y_true_encoded == y_pred):.2%}")
    print(f"Confusion matrix: {OUT_DIR / 'mlp_confusion_matrix.png'}")
    print(f"F1 graph: {OUT_DIR / 'mlp_f1_scores.png'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
