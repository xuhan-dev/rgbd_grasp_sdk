from __future__ import annotations

from pathlib import Path
from typing import Any

from rgbd_grasp_sdk.errors import BackendUnavailableError
from rgbd_grasp_sdk.grasping.rng_adapter_utils import rng_grasp_to_candidate
from rgbd_grasp_sdk.types import CameraIntrinsics, GraspPredictionResult, GraspRequest


class RngGraspPredictor:
    def __init__(self, options: dict[str, Any], predictor: Any | None = None):
        self.options = options
        self.checkpoint_path = options.get("checkpoint_path")
        if predictor is None and not self.checkpoint_path:
            raise BackendUnavailableError("RngGraspPredictor 需要 checkpoint_path")
        self.predictor = predictor or self._load_predictor()

    def predict(self, request: GraspRequest) -> GraspPredictionResult:
        visualize_3d = bool(request.metadata.get("visualize_3d", False))
        pred_gg, point_cloud = self.predictor.predict(
            request.rgb,
            request.depth,
            vis=visualize_3d,
        )
        candidates = []
        for grasp in pred_gg:
            candidates.append(
                rng_grasp_to_candidate(
                    grasp,
                    center_px=_infer_center_px(grasp, request.intrinsics),
                )
            )
        return GraspPredictionResult(
            candidates=candidates,
            point_cloud=point_cloud,
            metadata={"backend": "rng"},
        )

    def _load_predictor(self) -> Any:
        checkpoint_path = Path(str(self.checkpoint_path))
        if not checkpoint_path.exists():
            raise BackendUnavailableError(f"RNG checkpoint_path 不存在: {checkpoint_path}")
        try:
            from rgbd_grasp_sdk.grasping._rng_import import load_rng_predictor
        except ImportError as exc:
            raise BackendUnavailableError("无法导入 RNG 适配层依赖") from exc
        return load_rng_predictor(self.options)


def _infer_center_px(grasp: Any, intrinsics: CameraIntrinsics) -> tuple[int, int]:
    center = getattr(grasp, "center_px", None)
    if center is None:
        center = getattr(grasp, "center", None)
    if center is None:
        translation = getattr(grasp, "translation", None)
        if translation is None:
            return (0, 0)
        x, y, z = translation
        if z == 0:
            return (0, 0)
        return (
            int(round(float(x) * intrinsics.fx / float(z) + intrinsics.cx)),
            int(round(float(y) * intrinsics.fy / float(z) + intrinsics.cy)),
        )
    return (int(center[0]), int(center[1]))
