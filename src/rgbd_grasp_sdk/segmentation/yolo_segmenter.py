from __future__ import annotations

from typing import Any

import numpy as np

from rgbd_grasp_sdk.errors import BackendUnavailableError, ConfigError
from rgbd_grasp_sdk.types import MaskResult, SegmentationRequest, SegmentationResult


class YoloSegmenter:
    def __init__(self, options: dict[str, Any], model: Any | None = None):
        self.options = options
        self.model = model or self._load_model(options)

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        results = self.model(
            request.rgb,
            conf=float(self.options.get("confidence", 0.25)),
            iou=float(self.options.get("iou", 0.6)),
            verbose=False,
        )
        masks: list[MaskResult] = []
        for result in results:
            masks.extend(self._extract_masks(result, request.target))
        return SegmentationResult(masks=masks, metadata={"backend": "yolo"})

    def _load_model(self, options: dict[str, Any]) -> Any:
        model_path = options.get("model_path")
        if not model_path:
            raise ConfigError("YoloSegmenter 需要 model_path")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendUnavailableError("缺少 ultralytics，请安装 rgbd-grasp-sdk[yolo]") from exc
        return YOLO(model_path)

    def _extract_masks(self, result: Any, target: str) -> list[MaskResult]:
        if getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
            return []

        mask_data = _to_numpy(result.masks.data)
        class_ids = _to_numpy(result.boxes.cls).astype(int)
        confidences = _to_numpy(result.boxes.conf)
        names = result.names

        masks: list[MaskResult] = []
        for index, class_id in enumerate(class_ids):
            label = names[int(class_id)]
            if label != target:
                continue
            masks.append(
                MaskResult(
                    mask=mask_data[index].astype(bool),
                    score=float(confidences[index]),
                    label=label,
                    metadata={"class_id": int(class_id)},
                )
            )
        return masks


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
