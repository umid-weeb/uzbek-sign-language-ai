from __future__ import annotations

import numpy as np
import torch

from data.external.uzslr_isolated_dynamic_constants import (
    POINT_LANDMARKS,
)


def preprocess_raw_sequence(frames: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert raw MediaPipe Holistic frames to the checkpoint input format.

    ``frames`` must contain 32 frames with the source layout:
    face (468x3), pose (33x4), right hand (21x3), left hand (21x3).
    """
    x = torch.as_tensor(frames, dtype=torch.float32)
    if x.shape != (32, 1662):
        raise ValueError(f"32x1662 raw frames kutilgan, olindi: {tuple(x.shape)}")

    face = x[:, : 468 * 3].reshape(-1, 468, 3)
    pose_start = 468 * 3
    pose = x[:, pose_start : pose_start + 33 * 4].reshape(-1, 33, 4)[..., :3]
    hand_start = pose_start + 33 * 4
    right = x[:, hand_start : hand_start + 21 * 3].reshape(-1, 21, 3)
    left = x[:, hand_start + 21 * 3 :].reshape(-1, 21, 3)
    landmarks = torch.cat([face, pose, right, left], dim=1)
    selected = landmarks[:, POINT_LANDMARKS, :2]

    center = selected[:, POINT_LANDMARKS.index(17) : POINT_LANDMARKS.index(17) + 1]
    center = torch.nanmean(center, dim=(0, 1), keepdim=True)
    center = torch.nan_to_num(center, nan=0.5)
    normalized = selected - center
    std = normalized.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = torch.nan_to_num(normalized / std)

    velocity = torch.zeros_like(normalized)
    velocity[1:] = normalized[1:] - normalized[:-1]
    acceleration = torch.zeros_like(normalized)
    acceleration[:-2] = normalized[2:] - normalized[:-2]
    return torch.cat(
        [
            normalized.flatten(start_dim=1),
            velocity.flatten(start_dim=1),
            acceleration.flatten(start_dim=1),
        ],
        dim=-1,
    )
