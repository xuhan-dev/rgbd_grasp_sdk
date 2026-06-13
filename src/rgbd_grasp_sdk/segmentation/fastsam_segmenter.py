from __future__ import annotations

from typing import Any

from rgbd_grasp_sdk.errors import BackendUnavailableError, ConfigError
from rgbd_grasp_sdk.types import MaskResult, SegmentationRequest, SegmentationResult


class FastSamSegmenter:
    def __init__(self, options: dict[str, Any], backend: Any | None = None):
        self.options = options
        self.backend = backend or self._load_backend(options)

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        mask, preview = self.backend.segment_text(request.rgb, request.target)
        masks = []
        if mask is not None and mask.size > 0 and mask.any():
            masks.append(
                MaskResult(mask=mask.astype(bool), score=None, label=request.target)
            )
        return SegmentationResult(
            masks=masks,
            preview=preview,
            metadata={"backend": "fastsam"},
        )

    def _load_backend(self, options: dict[str, Any]) -> Any:
        model_path = options.get("model_path")
        if not model_path:
            raise ConfigError("FastSamSegmenter 需要 model_path")
        try:
            from rgbd_grasp_sdk.segmentation._fastsam_import import FastSamBackend
        except ImportError as exc:
            raise BackendUnavailableError("无法导入 FastSAM 适配层依赖") from exc
        return FastSamBackend(options)
