from __future__ import annotations

from typing import Protocol

from rgbd_grasp_sdk.types import SegmentationRequest, SegmentationResult


class Segmenter(Protocol):
    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        """根据 RGB 图像和目标描述返回分割结果。"""
