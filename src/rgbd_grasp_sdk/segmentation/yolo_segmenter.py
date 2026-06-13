from __future__ import annotations

from typing import Any

from rgbd_grasp_sdk.errors import BackendUnavailableError
from rgbd_grasp_sdk.types import SegmentationRequest, SegmentationResult


class YoloSegmenter:
    def __init__(self, options: dict[str, Any]):
        self.options = options

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        raise BackendUnavailableError("YoloSegmenter 尚未接入真实 YOLO 模型")
