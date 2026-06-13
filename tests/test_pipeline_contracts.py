import numpy as np

from rgbd_grasp_sdk.types import (
    CameraIntrinsics,
    GraspCandidate,
    PipelineResult,
    PipelineStatus,
    Pose6D,
)


def test_core_types_are_constructible():
    intrinsics = CameraIntrinsics(fx=600.0, fy=601.0, cx=320.0, cy=240.0)
    pose = Pose6D(x=0.1, y=0.2, z=0.3, roll=0.0, pitch=1.57, yaw=3.14)
    grasp = GraspCandidate(
        pose=pose,
        score=0.9,
        center_px=(100, 120),
        width=0.06,
        metadata={"source": "test"},
    )
    result = PipelineResult(
        status=PipelineStatus.SUCCESS,
        best_grasp=grasp,
        candidate_grasps=[grasp],
        target_mask=np.ones((4, 4), dtype=bool),
        timings={"total": 0.01},
        metadata={"target": "apple"},
    )

    assert intrinsics.matrix.shape == (3, 3)
    assert result.best_grasp is grasp
    assert result.status is PipelineStatus.SUCCESS


from rgbd_grasp_sdk.errors import BackendUnavailableError
from rgbd_grasp_sdk.grasping.factory import create_grasp_predictor
from rgbd_grasp_sdk.segmentation.factory import create_segmenter


def test_unknown_backends_raise_clear_errors():
    try:
        create_segmenter("unknown", {})
    except BackendUnavailableError as exc:
        assert "未知分割 backend" in str(exc)
    else:
        raise AssertionError("create_segmenter should fail for unknown backend")

    try:
        create_grasp_predictor("unknown", {})
    except BackendUnavailableError as exc:
        assert "未知抓取 backend" in str(exc)
    else:
        raise AssertionError("create_grasp_predictor should fail for unknown backend")


from rgbd_grasp_sdk.pipeline.grasp_pipeline import GraspPipeline
from rgbd_grasp_sdk.types import (
    GraspPredictionResult,
    GraspRequest,
    MaskResult,
    SegmentationRequest,
    SegmentationResult,
)


class MockSegmenter:
    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        mask = np.zeros(request.rgb.shape[:2], dtype=bool)
        mask[2, 3] = True
        return SegmentationResult(masks=[MaskResult(mask=mask, score=1.0, label=request.target)])


class MockGraspPredictor:
    def predict(self, request: GraspRequest) -> GraspPredictionResult:
        inside = GraspCandidate(
            pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            score=0.9,
            center_px=(3, 2),
        )
        outside = GraspCandidate(
            pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            score=1.0,
            center_px=(0, 0),
        )
        return GraspPredictionResult(candidates=[outside, inside])


def test_pipeline_uses_injected_interfaces_and_filters_by_mask():
    pipeline = GraspPipeline(
        segmenter=MockSegmenter(),
        grasp_predictor=MockGraspPredictor(),
    )
    rgb = np.zeros((6, 6, 3), dtype=np.uint8)
    depth = np.ones((6, 6), dtype=np.uint16)
    intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=3.0, cy=3.0)

    result = pipeline.run(rgb=rgb, depth=depth, intrinsics=intrinsics, target="apple")

    assert result.status is PipelineStatus.SUCCESS
    assert result.best_grasp is not None
    assert result.best_grasp.center_px == (3, 2)
    assert len(result.candidate_grasps) == 1
    assert "total" in result.timings


from rgbd_grasp_sdk.config.schema import MaskConfig


def test_pipeline_applies_mask_postprocess_config():
    class OnePixelSegmenter:
        def segment(self, request: SegmentationRequest) -> SegmentationResult:
            mask = np.zeros(request.rgb.shape[:2], dtype=bool)
            mask[2, 2] = True
            return SegmentationResult(masks=[MaskResult(mask=mask)])

    class NearbyGraspPredictor:
        def predict(self, request: GraspRequest) -> GraspPredictionResult:
            return GraspPredictionResult(
                candidates=[
                    GraspCandidate(
                        pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
                        score=0.8,
                        center_px=(3, 2),
                    )
                ]
            )

    pipeline = GraspPipeline(
        segmenter=OnePixelSegmenter(),
        grasp_predictor=NearbyGraspPredictor(),
        mask_config=MaskConfig(dilate_kernel=3, dilate_iterations=1),
    )

    result = pipeline.run(
        rgb=np.zeros((6, 6, 3), dtype=np.uint8),
        depth=np.ones((6, 6), dtype=np.uint16),
        intrinsics=CameraIntrinsics(fx=600.0, fy=600.0, cx=3.0, cy=3.0),
        target="apple",
    )

    assert result.status is PipelineStatus.SUCCESS
    assert result.best_grasp is not None
