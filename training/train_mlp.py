
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/processed/train.csv"
TEST_PATH = ROOT / "data/processed/test.csv"
MODEL_DIR = ROOT / "models"
REPORT_PATH = ROOT / "data/processed/metrics_mlp.txt"

SEED = 42
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def load_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames

        if not columns or "label" not in columns:
            raise ValueError(f"label ustuni topilmadi: {path}")

        features = [c for c in columns if c != "label"]
        X, y = [], []

        for row in reader:
            X.append([float(row[c]) for c in features])
            y.append(row["label"])

    if not X:
        raise ValueError(f"Dataset bo'sh: {path}")

    return np.asarray(X, dtype=np.float32), np.asarray(y), features


class GestureMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def main():
    X_train, y_train, train_columns = load_csv(TRAIN_PATH)
    X_test, y_test, test_columns = load_csv(TEST_PATH)

    if train_columns != test_columns:
        raise ValueError("Train va test ustunlari mos emas.")

    if set(y_test) - set(y_train):
        raise ValueError("Testda train tarkibida bo'lmagan klass bor.")

    # Klasslarni sonli indekslarga aylantiramiz.
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    # Scaler faqat train ma'lumotida fit qilinadi.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train_encoded.astype(np.int64)),
    )

    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = GestureMLP(
        input_size=X_train.shape[1],
        num_classes=len(encoder.classes_),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Device: {DEVICE}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(encoder.classes_)}")
    print("\nTraining boshlandi...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        average_loss = total_loss / len(train_dataset)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d}/{EPOCHS} "
                f"| train loss: {average_loss:.4f}"
            )

    # Test faqat yakuniy baholash uchun ishlatiladi.
    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test).to(DEVICE)
        logits = model(test_tensor)
        predictions = logits.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test_encoded, predictions)
    labels = np.arange(len(encoder.classes_))

    report = classification_report(
        y_test_encoded,
        predictions,
        labels=labels,
        target_names=encoder.classes_,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test_encoded,
        predictions,
        labels=labels,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "input_size": X_train.shape[1],
            "labels": encoder.classes_.tolist(),
            "feature_columns": train_columns,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "seed": SEED,
        },
        MODEL_DIR / "gesture_mlp.pt",
    )

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Accuracy: {accuracy:.4%}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\n\nConfusion matrix:\n")
        f.write(str(matrix))

    print(f"\nMLP test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(report)
    print(f"\nModel: {MODEL_DIR / 'gesture_mlp.pt'}")
    print(f"Hisobot: {REPORT_PATH}")


if __name__ == "__main__":
    main()
