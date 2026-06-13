from __future__ import annotations

from typing import Any

from rgbd_grasp_sdk.errors import BackendUnavailableError
from rgbd_grasp_sdk.segmentation.base import Segmenter


def create_segmenter(backend: str, options: dict[str, Any]) -> Segmenter:
    if backend == "yolo":
        from rgbd_grasp_sdk.segmentation.yolo_segmenter import YoloSegmenter

        return YoloSegmenter(options)
    if backend == "fastsam":
        from rgbd_grasp_sdk.segmentation.fastsam_segmenter import FastSamSegmenter

        return FastSamSegmenter(options)
    raise BackendUnavailableError(f"未知分割 backend: {backend}")
