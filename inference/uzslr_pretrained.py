from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch

from inference.uzslr_preprocess import preprocess_raw_sequence

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "external" / "uzslr-isolated-dynamic"
CHECKPOINT_PATH = SOURCE_DIR / "best_model.pth"
MODEL_SOURCE = SOURCE_DIR / "inference03_model.py"
CONFIG_SOURCE = SOURCE_DIR / "inference01_config.py"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Source module yuklanmadi: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pretrained_uzslr() -> tuple[torch.nn.Module, list[str]]:
    """Load the public Uzbek 50-sign CNN/Transformer checkpoint."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint topilmadi: {CHECKPOINT_PATH}")
    model_source = _load_module(MODEL_SOURCE, "public_uzslr_model")
    config = _load_module(CONFIG_SOURCE, "public_uzslr_config")
    model = model_source.SignLanguageModel(
        max_len=config.MAX_LEN,
        channels=config.CHANNELS,
        num_classes=config.NUM_CLASSES,
        dim=config.MODEL_DIM,
    )
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, list(config.DEFAULT_SIGNS)


def predict_pretrained(
    model: torch.nn.Module,
    batch: torch.Tensor,
    labels: list[str],
) -> tuple[str, float]:
    """Predict one preprocessed [1, 32, 708] sequence."""
    if batch.shape != (1, 32, 708):
        raise ValueError("Pretrained model [1, 32, 708] input kutadi.")
    if len(labels) != 50:
        raise ValueError("Pretrained model uchun 50 ta label kerak.")
    with torch.no_grad():
        probabilities = model(batch).softmax(dim=-1)[0]
    index = int(probabilities.argmax())
    return labels[index], float(probabilities[index])


def predict_raw_sequence(
    model: torch.nn.Module,
    raw_frames: torch.Tensor,
    labels: list[str],
) -> tuple[str, float]:
    """Preprocess and predict a raw 32-frame Holistic sequence."""
    batch = preprocess_raw_sequence(raw_frames).unsqueeze(0)
    return predict_pretrained(model, batch, labels)
