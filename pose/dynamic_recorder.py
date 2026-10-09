from __future__ import annotations

import argparse
import csv
import ssl
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.temporal_features import FEATURE_VERSION, frame_to_features

ssl._create_default_https_context = ssl._create_unverified_context

SEQUENCE_DIR = ROOT / "data" / "dynamic" / "sequences"
METADATA_PATH = ROOT / "data" / "dynamic" / "metadata.csv"


def save_sequence(label: str, signer: str, view: str, frames: list[np.ndarray]) -> Path:
    if len(frames) < 8:
        raise ValueError("Sequence juda qisqa: kamida 8 ta valid frame kerak.")

    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    sequence_id = f"seq_{int(time.time() * 1000)}"
    path = SEQUENCE_DIR / f"{sequence_id}.npz"
    np.savez_compressed(path, features=np.stack(frames), feature_version=FEATURE_VERSION)

    has_header = METADATA_PATH.exists()
    with METADATA_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sequence_id", "label", "signer_id", "view", "frame_count"],
        )
        if not has_header:
            writer.writeheader()
        writer.writerow(
            {
                "sequence_id": sequence_id,
                "label": label,
                "signer_id": signer,
                "view": view,
                "frame_count": len(frames),
            }
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record dynamic sign sequences.")
    parser.add_argument("--label", required=True, help="Gesture yoki so'z labeli")
    parser.add_argument("--signer", default="signer_01")
    parser.add_argument("--view", default="front", choices=["front", "left", "right"])
    args = parser.parse_args()

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

    frames: list[np.ndarray] = []
    recording = False
    print("SPACE=start/stop, Q=quit")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            keypoints, scores = pose(frame)
            display = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.3)

            valid = len(keypoints) > 0
            if valid:
                features = frame_to_features(keypoints[0], scores[0])
                if recording:
                    frames.append(features)

            status = f"{args.label} | {'REC' if recording else 'READY'} | {len(frames)}"
            cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Dynamic recorder", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and not recording:
                frames = []
                recording = True
            elif key == ord(" ") and recording:
                recording = False
                if frames:
                    path = save_sequence(args.label, args.signer, args.view, frames)
                    print(f"Saved: {path}")
                    frames = []
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
