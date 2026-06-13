from __future__ import annotations

import numpy as np

from rgbd_grasp_sdk.config.schema import MaskConfig
from rgbd_grasp_sdk.filtering.mask_filter import filter_grasps_by_mask
from rgbd_grasp_sdk.grasping.base import GraspPredictor
from rgbd_grasp_sdk.masks.postprocess import postprocess_masks
from rgbd_grasp_sdk.ranking.base import GraspRanker
from rgbd_grasp_sdk.ranking.default_ranker import DefaultGraspRanker
from rgbd_grasp_sdk.segmentation.base import Segmenter
from rgbd_grasp_sdk.timing import TimingRecorder
from rgbd_grasp_sdk.types import (
    CameraIntrinsics,
    GraspRequest,
    PipelineError,
    PipelineResult,
    PipelineStatus,
    SegmentationRequest,
)


class GraspPipeline:
    def __init__(
        self,
        segmenter: Segmenter,
        grasp_predictor: GraspPredictor,
        ranker: GraspRanker | None = None,
        mask_config: MaskConfig | None = None,
        min_grasp_score: float = 0.0,
        visualize_3d: bool = False,
    ) -> None:
        self.segmenter = segmenter
        self.grasp_predictor = grasp_predictor
        self.ranker = ranker or DefaultGraspRanker()
        self.mask_config = mask_config or MaskConfig()
        self.min_grasp_score = min_grasp_score
        self.visualize_3d = visualize_3d

    def run(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        target: str,
    ) -> PipelineResult:
        timer = TimingRecorder()

        with timer.measure("total"):
            if rgb.shape[:2] != depth.shape[:2]:
                return PipelineResult(
                    status=PipelineStatus.FAILED,
                    timings=timer.timings,
                    metadata={"target": target},
                    error=PipelineError(
                        code="shape_mismatch",
                        message="RGB 和 depth 尺寸不匹配",
                        detail={"rgb_shape": rgb.shape, "depth_shape": depth.shape},
                    ),
                )

            with timer.measure("segmentation"):
                segmentation = self.segmenter.segment(
                    SegmentationRequest(rgb=rgb, target=target)
                )
            target_mask = postprocess_masks(segmentation.masks, self.mask_config)
            if target_mask.size == 0 or not bool(target_mask.any()):
                return PipelineResult(
                    status=PipelineStatus.FAILED,
                    target_mask=target_mask,
                    segmentation_preview=segmentation.preview,
                    timings=timer.timings,
                    metadata={"target": target},
                    error=PipelineError(code="empty_mask", message="目标 mask 为空"),
                )

            with timer.measure("grasping"):
                prediction = self.grasp_predictor.predict(
                    GraspRequest(
                        rgb=rgb,
                        depth=depth,
                        intrinsics=intrinsics,
                        target_mask=target_mask,
                        metadata={"visualize_3d": self.visualize_3d},
                    )
                )

            with timer.measure("filtering"):
                filtered = filter_grasps_by_mask(
                    prediction.candidates,
                    target_mask,
                    min_score=self.min_grasp_score,
                )
            if not filtered:
                return PipelineResult(
                    status=PipelineStatus.FAILED,
                    target_mask=target_mask,
                    point_cloud=prediction.point_cloud,
                    segmentation_preview=segmentation.preview,
                    timings=timer.timings,
                    metadata={"target": target},
                    error=PipelineError(code="no_valid_grasp", message="没有有效抓取候选"),
                )

            with timer.measure("ranking"):
                ranked = self.ranker.rank(filtered)

        return PipelineResult(
            status=PipelineStatus.SUCCESS,
            best_grasp=ranked[0],
            candidate_grasps=ranked,
            target_mask=target_mask,
            point_cloud=prediction.point_cloud,
            segmentation_preview=segmentation.preview,
            timings=timer.timings,
            metadata={"target": target},
        )
