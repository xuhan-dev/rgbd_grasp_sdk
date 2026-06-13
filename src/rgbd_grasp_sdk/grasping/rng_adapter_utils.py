from __future__ import annotations

from typing import Any

import numpy as np

from rgbd_grasp_sdk.types import GraspCandidate, Pose6D


def rng_grasp_to_candidate(rng_grasp: Any, center_px: tuple[int, int]) -> GraspCandidate:
    translation = np.asarray(rng_grasp.translation, dtype=float)
    roll, pitch, yaw = _rotation_matrix_to_euler_xyz(
        np.asarray(rng_grasp.rotation, dtype=float)
    )
    return GraspCandidate(
        pose=Pose6D(
            x=float(translation[0]),
            y=float(translation[1]),
            z=float(translation[2]),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        ),
        score=float(getattr(rng_grasp, "score", 0.0)),
        center_px=center_px,
        width=float(getattr(rng_grasp, "width", 0.0)),
        metadata={"source": "rng"},
    )


def _rotation_matrix_to_euler_xyz(matrix: np.ndarray) -> tuple[float, float, float]:
    sy = float(np.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        pitch = float(np.arctan2(-matrix[2, 0], sy))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        roll = float(np.arctan2(-matrix[1, 2], matrix[1, 1]))
        pitch = float(np.arctan2(-matrix[2, 0], sy))
        yaw = 0.0
    return roll, pitch, yaw
