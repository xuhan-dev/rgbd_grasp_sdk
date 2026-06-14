from __future__ import annotations

from typing import Protocol

from rgbd_grasp_sdk.types import PipelineResult


class GraspPublisher(Protocol):
    def publish(self, result: PipelineResult) -> None:
        ...
