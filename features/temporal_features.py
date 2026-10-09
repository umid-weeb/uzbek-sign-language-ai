from __future__ import annotations

import numpy as np

LEFT_HAND_SLICE = slice(92, 113)
RIGHT_HAND_SLICE = slice(113, 134)
FEATURE_VERSION = "rtmlib-hands-v1"


def _hand_features(
    keypoints: np.ndarray,
    scores: np.ndarray,
    hand_slice: slice,
) -> np.ndarray:
    points = keypoints[hand_slice]
    confidence = scores[hand_slice]

    wrist = points[0]
    local = points - wrist
    palm_scale = np.linalg.norm(points[9] - wrist)
    if palm_scale < 1e-6:
        palm_scale = 1.0
    local = local / palm_scale

    return np.concatenate(
        [local.astype(np.float32).reshape(-1), confidence.astype(np.float32)]
    )


def frame_to_features(
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Convert one RTMLib whole-body result into a fixed-size two-hand vector."""
    left = _hand_features(keypoints, scores, LEFT_HAND_SLICE)
    right = _hand_features(keypoints, scores, RIGHT_HAND_SLICE)
    return np.concatenate([left, right]).astype(np.float32)


def add_motion_features(sequence: np.ndarray) -> np.ndarray:
    """Append velocity and acceleration to a [time, feature] sequence."""
    if sequence.ndim != 2 or sequence.shape[0] < 1:
        raise ValueError("Sequence [time, feature] formatida bo'lishi kerak.")

    velocity = np.zeros_like(sequence)
    if len(sequence) > 1:
        velocity[1:] = np.diff(sequence, axis=0)

    acceleration = np.zeros_like(sequence)
    if len(sequence) > 2:
        acceleration[2:] = np.diff(velocity, axis=0)[1:]

    return np.concatenate([sequence, velocity, acceleration], axis=1)
