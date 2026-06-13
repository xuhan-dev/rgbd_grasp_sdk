from __future__ import annotations

import numpy as np

from rgbd_grasp_sdk.types import GraspCandidate


def filter_grasps_by_mask(
    candidates: list[GraspCandidate],
    mask: np.ndarray,
    min_score: float = 0.0,
) -> list[GraspCandidate]:
    if mask.size == 0:
        return []

    height, width = mask.shape[:2]
    filtered: list[GraspCandidate] = []
    for candidate in candidates:
        x, y = candidate.center_px
        if candidate.score < min_score:
            continue
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        if bool(mask[y, x]):
            filtered.append(candidate)
    return filtered
