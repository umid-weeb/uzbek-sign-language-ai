from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from features.temporal_features import add_motion_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "dynamic"
DEFAULT_OUTPUT = ROOT / "models" / "dynamic_transformer.pt"
SEED = 42


class SequenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], sequence_dir: Path, labels: dict[str, int]):
        self.rows = rows
        self.sequence_dir = sequence_dir
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = self.sequence_dir / f"{row['sequence_id']}.npz"
        data = np.load(path)
        raw_features = data["features"].astype(np.float32)
        features = torch.from_numpy(add_motion_features(raw_features))
        return features, self.labels[row["label"]]


def collate_batch(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    mask = torch.arange(padded.shape[1])[None, :] >= lengths[:, None]
    return padded, mask, torch.tensor(labels, dtype=torch.long)


class DynamicTransformer(nn.Module):
    def __init__(self, input_size: int, class_count: int, hidden_size: int = 128):
        super().__init__()
        self.projection = nn.Linear(input_size, hidden_size)
        position = torch.arange(512, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-np.log(10000.0) / hidden_size)
        )
        positional = torch.zeros(512, hidden_size)
        positional[:, 0::2] = torch.sin(position * div_term)
        positional[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional", positional.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=0.15,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.classifier = nn.Linear(hidden_size, class_count)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        projected = self.projection(features)
        if projected.shape[1] > self.positional.shape[1]:
            raise ValueError("Sequence 512 frame limitidan uzun.")
        encoded = self.encoder(
            projected + self.positional[:, : projected.shape[1]],
            src_key_padding_mask=padding_mask,
        )
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)


def load_rows(data_dir: Path) -> list[dict[str, str]]:
    path = data_dir / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Metadata topilmadi: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Dynamic metadata bo'sh.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rows = load_rows(args.data)
    labels = {label: index for index, label in enumerate(sorted({row["label"] for row in rows}))}
    signer_ids = sorted({row["signer_id"] for row in rows})
    if len(signer_ids) < 2:
        raise ValueError("Signer-based test uchun kamida 2 ta signer kerak.")
    random.shuffle(signer_ids)
    test_signers = set(signer_ids[: max(1, len(signer_ids) // 5)])
    test_rows = [row for row in rows if row["signer_id"] in test_signers]
    train_rows = [row for row in rows if row["signer_id"] not in test_signers]
    if not train_rows or not test_rows:
        raise ValueError("Signer-based train/test split bo'sh chiqdi.")
    train_labels = {row["label"] for row in train_rows}
    missing_labels = sorted(set(labels) - train_labels)
    if missing_labels:
        raise ValueError(
            "Train splitda quyidagi klasslar yo'q: "
            + ", ".join(missing_labels)
            + ". Har bir klassni kamida ikki signer bilan yozing."
        )
    train = SequenceDataset(train_rows, args.data / "sequences", labels)
    test = SequenceDataset(test_rows, args.data / "sequences", labels)
    loader = DataLoader(train, batch_size=16, shuffle=True, collate_fn=collate_batch)
    model = DynamicTransformer(input_size=378, class_count=len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for features, mask, target in loader:
            optimizer.zero_grad()
            loss = criterion(model(features, mask), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:03d}/{args.epochs} loss={total / max(1, len(loader)):.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for features, mask, target in DataLoader(test, batch_size=16, collate_fn=collate_batch):
            correct += int((model(features, mask).argmax(1) == target).sum())
    accuracy = correct / max(1, len(test))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": 378,
            "labels": labels,
            "feature_version": "rtmlib-hands-v1",
            "accuracy": accuracy,
        },
        args.output,
    )
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Model saved: {args.output}")


if __name__ == "__main__":
    main()
