from __future__ import annotations

from typing import Protocol

from rgbd_grasp_sdk.types import GraspPredictionResult, GraspRequest


class GraspPredictor(Protocol):
    def predict(self, request: GraspRequest) -> GraspPredictionResult:
        """根据 RGB-D、内参和目标 mask 返回抓取候选。"""
