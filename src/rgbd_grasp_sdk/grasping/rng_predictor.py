from __future__ import annotations

from typing import Any

from rgbd_grasp_sdk.errors import BackendUnavailableError
from rgbd_grasp_sdk.types import GraspPredictionResult, GraspRequest


class RngGraspPredictor:
    def __init__(self, options: dict[str, Any]):
        self.options = options

    def predict(self, request: GraspRequest) -> GraspPredictionResult:
        raise BackendUnavailableError("RngGraspPredictor 尚未接入真实 RegionNormalizedGrasp 模型")
