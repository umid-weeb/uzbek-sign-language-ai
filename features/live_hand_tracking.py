from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEFT_HAND_RANGE = slice(91, 112)
RIGHT_HAND_RANGE = slice(112, 133)


@dataclass
class SmoothedHand:
    keypoints: np.ndarray | None = None
    scores: np.ndarray | None = None
    missing_frames: int = 0


class HandLandmarkSmoother:
    """Confidence-aware adaptive EMA for one tracked hand."""

    def __init__(
        self,
        *,
        min_score: float = 0.05,
        max_missing_frames: int = 3,
        slow_alpha: float = 0.18,
        fast_alpha: float = 0.65,
    ) -> None:
        self.min_score = min_score
        self.max_missing_frames = max_missing_frames
        self.slow_alpha = slow_alpha
        self.fast_alpha = fast_alpha
        self.state = SmoothedHand()

    def update(
        self,
        keypoints: np.ndarray | None,
        scores: np.ndarray | None,
    ) -> SmoothedHand:
        valid = (
            keypoints is not None
            and scores is not None
            and keypoints.shape == (21, 2)
            and scores.shape == (21,)
            and float(np.min(scores)) >= self.min_score
        )
        if not valid:
            self.state.missing_frames += 1
            if self.state.missing_frames > self.max_missing_frames:
                self.state.keypoints = None
                self.state.scores = None
            return self.state

        points = np.asarray(keypoints, dtype=np.float32)
        point_scores = np.asarray(scores, dtype=np.float32)
        if self.state.keypoints is None:
            self.state.keypoints = points.copy()
        else:
            movement = float(np.linalg.norm(points - self.state.keypoints, axis=1).mean())
            scale = max(float(np.linalg.norm(points[9] - points[0])), 1.0)
            normalized_movement = min(movement / scale, 1.0)
            alpha = self.slow_alpha + (self.fast_alpha - self.slow_alpha) * normalized_movement
            self.state.keypoints = (
                (1.0 - alpha) * self.state.keypoints + alpha * points
            ).astype(np.float32)
        self.state.scores = point_scores
        self.state.missing_frames = 0
        return self.state


def choose_best_hand(
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Choose the highest-confidence visible hand from RTMLib's 133 layout."""
    if keypoints.shape != (133, 2) or scores.shape != (133,):
        raise ValueError("RTMLib Wholebody 133-keypoint layout kutilgan.")
    candidates = (
        ("left", keypoints[LEFT_HAND_RANGE], scores[LEFT_HAND_RANGE]),
        ("right", keypoints[RIGHT_HAND_RANGE], scores[RIGHT_HAND_RANGE]),
    )
    name, points, point_scores = max(candidates, key=lambda item: float(np.mean(item[2])))
    if float(np.mean(point_scores)) < 0.05:
        return None, None, None
    return points, point_scores, name


def extract_wrist_relative_features(hand_keypoints: np.ndarray) -> np.ndarray:
    if hand_keypoints.shape != (21, 2):
        raise ValueError("21x2 hand keypoints kutilgan.")
    wrist = hand_keypoints[0]
    features = (hand_keypoints - wrist).astype(np.float32)
    return np.round(features, 4).reshape(-1)


def extract_bounding_box_features(
    hand_keypoints: np.ndarray,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """Match sign2text's MediaPipe feature contract from RTMLib points."""
    if hand_keypoints.shape != (21, 2):
        raise ValueError("21x2 hand keypoints kutilgan.")
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("Frame o'lchami musbat bo'lishi kerak.")
    points = hand_keypoints.astype(np.float32).copy()
    points[:, 0] /= float(width)
    points[:, 1] /= float(height)
    points -= points.min(axis=0)
    return np.round(points, 4).reshape(-1)


def validate_rtmlib_output(
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Reject silently-corrupted detector output before feature extraction."""
    if keypoints.shape != (133, 2) or scores.shape != (133,):
        raise ValueError(
            "RTMLib output 133x2 keypoints va 133 scores bo'lishi kerak: "
            f"{keypoints.shape}, {scores.shape}"
        )
