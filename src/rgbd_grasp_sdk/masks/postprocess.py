from __future__ import annotations

import cv2
import numpy as np

from rgbd_grasp_sdk.config.schema import MaskConfig
from rgbd_grasp_sdk.types import MaskResult


def merge_masks(masks: list[MaskResult]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)

    merged = np.zeros_like(masks[0].mask, dtype=bool)
    for item in masks:
        merged |= item.mask.astype(bool)
    return merged


def postprocess_masks(masks: list[MaskResult], config: MaskConfig) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)

    if config.merge_instances:
        result = merge_masks(masks)
    else:
        result = masks[0].mask.astype(bool)

    if config.min_area > 0:
        result = _remove_small_components(result, config.min_area)

    if config.dilate_kernel > 0 and config.dilate_iterations > 0:
        kernel = np.ones((config.dilate_kernel, config.dilate_kernel), dtype=np.uint8)
        result = cv2.dilate(
            result.astype(np.uint8),
            kernel,
            iterations=config.dilate_iterations,
        ).astype(bool)

    return result


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    cleaned = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = True
    return cleaned
