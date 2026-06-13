from __future__ import annotations

import numpy as np

from rgbd_grasp_sdk.types import MaskResult


def merge_masks(masks: list[MaskResult]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)

    merged = np.zeros_like(masks[0].mask, dtype=bool)
    for item in masks:
        merged |= item.mask.astype(bool)
    return merged
