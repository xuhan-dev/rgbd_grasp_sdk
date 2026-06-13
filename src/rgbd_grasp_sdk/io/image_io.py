from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rgbd_grasp_sdk.errors import InputValidationError


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise InputValidationError(f"无法读取 RGB 图像: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_depth(path: str | Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise InputValidationError(f"无法读取 depth 图像: {path}")
    if depth.ndim != 2:
        raise InputValidationError(f"depth 图像必须是单通道: {path}")
    return depth
