from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Wholebody

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.live_hand_tracking import extract_wrist_relative_features
from pose.camera_utils import open_camera, list_cameras

LABELS = [
    "A", "B", "CH", "D", "E", "F", "G", "G_star", "H", "I", "J", "K",
    "L", "M", "N", "NG", "O", "O_star", "P", "Q", "R", "S", "SH", "T",
    "U", "V", "X", "Y", "Z",
]
OUTPUT = ROOT / "data" / "camera_calibration.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RTMLib camera calibration landmarks.")
    parser.add_argument("--samples-per-label", type=int, default=100)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--list-cameras", action="store_true")
    args = parser.parse_args()
    if args.list_cameras:
        print("Mavjud camera indexlar:", list_cameras())
        return
    if args.samples_per_label < 20:
        raise ValueError("Har bir label uchun kamida 20 sample kerak.")

    detector = Wholebody(
        mode="lightweight", to_openpose=False, backend="onnxruntime", device="cpu"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label"] + [
        f"{axis}{i}" for i in range(21) for axis in ("x", "y")
    ]
    file_exists = args.output.exists() and args.output.stat().st_size > 0
    output = args.output.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    cap = open_camera(args.camera, 640, 480)
    print(f"Camera index: {args.camera}")

    index = 0
    captured = 0
    collecting = False
    print("S=collect, N=next label, Q=quit. Har bir harfni turli masofa va burchakda ko'rsating.")
    try:
        while index < len(LABELS):
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(cv2.resize(frame, (640, 480)), 1)
            keypoints, scores = detector(frame)
            label = LABELS[index]
            hand = None
            hand_scores = None
            if len(keypoints):
                candidates = [
                    (keypoints[0, 91:112], scores[0, 91:112]),
                    (keypoints[0, 112:133], scores[0, 112:133]),
                ]
                hand, hand_scores = max(candidates, key=lambda item: float(np.mean(item[1])))
            valid = hand is not None and hand_scores is not None and float(np.min(hand_scores)) >= 0.10
            if collecting and valid and captured < args.samples_per_label:
                features = extract_wrist_relative_features(hand)
                writer.writerow({"label": label, **{
                    f"{axis}{i}": float(features[2 * i + offset])
                    for i in range(21) for axis, offset in (("x", 0), ("y", 1))
                }})
                output.flush()
                captured += 1
            text = f"{label}: {captured}/{args.samples_per_label} | S collect N next Q quit"
            cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            cv2.putText(frame, "HAND OK" if valid else "HAND NOT READY", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Camera calibration", frame)
            pressed = cv2.waitKey(1) & 0xFF
            if pressed == ord("q"):
                break
            if pressed == ord("s"):
                collecting = True
            if pressed == ord("n") or captured >= args.samples_per_label:
                index += 1
                captured = 0
                collecting = False
    finally:
        output.close()
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saqlangan calibration CSV: {args.output}")


if __name__ == "__main__":
    main()
