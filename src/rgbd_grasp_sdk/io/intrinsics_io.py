from __future__ import annotations

from pathlib import Path

import numpy as np

from rgbd_grasp_sdk.errors import InputValidationError
from rgbd_grasp_sdk.types import CameraIntrinsics


def read_intrinsics_npz(path: str | Path) -> CameraIntrinsics:
    data = np.load(path)
    if "K" not in data:
        raise InputValidationError(f"内参 npz 必须包含 K: {path}")
    matrix = np.asarray(data["K"], dtype=np.float32)
    if matrix.shape != (3, 3):
        raise InputValidationError(f"K 必须是 3x3 矩阵: {path}")
    return CameraIntrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
    )
