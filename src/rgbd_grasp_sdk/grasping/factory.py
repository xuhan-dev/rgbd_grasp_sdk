from __future__ import annotations

from typing import Any

from rgbd_grasp_sdk.errors import BackendUnavailableError
from rgbd_grasp_sdk.grasping.base import GraspPredictor


def create_grasp_predictor(backend: str, options: dict[str, Any]) -> GraspPredictor:
    if backend == "rng":
        from rgbd_grasp_sdk.grasping.rng_predictor import RngGraspPredictor

        return RngGraspPredictor(options)
    raise BackendUnavailableError(f"未知抓取 backend: {backend}")
