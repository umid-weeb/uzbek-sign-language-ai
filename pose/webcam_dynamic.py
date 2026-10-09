from __future__ import annotations

import argparse
import ssl
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from rtmlib import Wholebody, draw_skeleton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.temporal_features import add_motion_features, frame_to_features
from training.train_dynamic import DynamicTransformer

ssl._create_default_https_context = ssl._create_unverified_context

DEFAULT_MODEL = ROOT / "models" / "dynamic_transformer.pt"


def load_model(path: Path) -> tuple[DynamicTransformer, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dynamic model topilmadi: {path}. Avval training/train_dynamic.py ni ishlating."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    labels_by_id = checkpoint["labels"]
    if isinstance(labels_by_id, dict):
        labels = [label for label, _ in sorted(labels_by_id.items(), key=lambda item: item[1])]
    else:
        labels = list(labels_by_id)
    model = DynamicTransformer(
        input_size=int(checkpoint["input_size"]),
        class_count=len(labels),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dynamic sign recognition.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--process-every", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    if args.window < 8 or args.window > 512:
        raise ValueError("--window 8 va 512 oralig'ida bo'lishi kerak.")
    if args.process_every < 1:
        raise ValueError("--process-every musbat bo'lishi kerak.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold 0 va 1 oralig'ida bo'lishi kerak.")

    model, labels = load_model(args.model)
    pose = Wholebody(
        mode="lightweight",
        to_openpose=False,
        backend="onnxruntime",
        device="cpu",
    )
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError("Kamera ochilmadi.")

    sequence: deque[np.ndarray] = deque(maxlen=args.window)
    predictions: deque[str] = deque(maxlen=5)
    frame_number = 0
    last_keypoints = None
    last_scores = None
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame_number += 1
            if frame_number % args.process_every == 0 or last_keypoints is None:
                keypoints, scores = pose(frame)
                last_keypoints = keypoints
                last_scores = scores
            keypoints = last_keypoints
            scores = last_scores
            display = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.3)
            text = "Sequence yig'ilmoqda..."
            if len(keypoints) > 0:
                sequence.append(frame_to_features(keypoints[0], scores[0]))
            if len(sequence) >= 8:
                raw = np.stack(sequence).astype(np.float32)
                features = torch.from_numpy(add_motion_features(raw)).unsqueeze(0)
                padding = torch.zeros((1, len(sequence)), dtype=torch.bool)
                with torch.no_grad():
                    probabilities = model(features, padding).softmax(dim=1)[0]
                index = int(probabilities.argmax())
                confidence = float(probabilities[index])
                if confidence >= args.threshold:
                    predictions.append(labels[index])
                    stable = Counter(predictions).most_common(1)[0][0]
                    text = f"{stable} ({confidence:.0%})"
                else:
                    text = f"Noaniq ({confidence:.0%})"
            cv2.putText(
                display,
                text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Dynamic Sign Language AI", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
