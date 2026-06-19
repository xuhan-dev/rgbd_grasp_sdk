from __future__ import annotations

import numpy as np
import pytest

from rgbd_grasp_sdk import RGBDGrasp
from rgbd_grasp_sdk.config.schema import (
    GraspingConfig,
    MaskConfig,
    OutputConfig,
    RankingConfig,
    SdkConfig,
    SegmentationConfig,
)
from rgbd_grasp_sdk.errors import InputValidationError
from rgbd_grasp_sdk.types import (
    CameraIntrinsics,
    GraspCandidate,
    PipelineResult,
    PipelineStatus,
    Pose6D,
)


class FakePipeline:
    def __init__(self):
        self.calls = []

    def run(self, rgb, depth, intrinsics, target):
        self.calls.append((rgb, depth, intrinsics, target))
        if target == "raise":
            raise RuntimeError("boom")
        if target == "fail":
            return PipelineResult(status=PipelineStatus.FAILED)
        grasp = GraspCandidate(
            pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            score=0.9,
            center_px=(1, 1),
        )
        return PipelineResult(
            status=PipelineStatus.SUCCESS,
            best_grasp=grasp,
            candidate_grasps=[grasp],
            timings={"total": 0.1},
        )


def _config() -> SdkConfig:
    return SdkConfig(
        segmentation=SegmentationConfig(
            backend="mock_seg",
            options={"model_path": "seg.pt", "device": "cpu"},
        ),
        grasping=GraspingConfig(
            backend="mock_grasp",
            options={"checkpoint_path": "rng.ckpt", "device": "cpu"},
        ),
        mask=MaskConfig(),
        ranking=RankingConfig(backend="default"),
        outputs=OutputConfig(visualize_3d=False),
    )


def _model_with_fake_pipeline():
    fake = FakePipeline()
    model = RGBDGrasp(_config(), pipeline_builder=lambda config, visualize_3d=None: fake)
    return model, fake


def test_rgbd_grasp_is_exported_from_package():
    assert RGBDGrasp.__name__ == "RGBDGrasp"


def test_predict_one_accepts_array_inputs_and_returns_pipeline_result():
    model, fake = _model_with_fake_pipeline()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.uint16)
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)

    result = model.predict_one(rgb=rgb, depth=depth, intrinsics=intrinsics, target="apple")

    assert result.status is PipelineStatus.SUCCESS
    assert fake.calls == [(rgb, depth, intrinsics, "apple")]


def test_predict_returns_list_for_single_keyword_input():
    model, _ = _model_with_fake_pipeline()

    results = model.predict(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0),
        target="apple",
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].status is PipelineStatus.SUCCESS


def test_predict_processes_source_list_independently():
    model, fake = _model_with_fake_pipeline()
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    source = [
        {
            "id": "a",
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16),
            "intrinsics": intrinsics,
            "target": "apple",
        },
        {
            "id": "b",
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16),
            "intrinsics": intrinsics,
            "target": "fail",
        },
    ]

    results = model.predict(source=source)

    assert [item.status for item in results] == [
        PipelineStatus.SUCCESS,
        PipelineStatus.FAILED,
    ]
    assert len(fake.calls) == 2


def test_strict_true_reraises_exceptions():
    model, _ = _model_with_fake_pipeline()

    with pytest.raises(RuntimeError, match="boom"):
        model.predict_one(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=np.ones((2, 2), dtype=np.uint16),
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0),
            target="raise",
            strict=True,
        )


def test_strict_false_converts_exceptions_to_failed_result():
    model, _ = _model_with_fake_pipeline()

    result = model.predict_one(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0),
        target="raise",
        strict=False,
    )

    assert result.status is PipelineStatus.FAILED
    assert result.error is not None
    assert result.error.code == "runtime_error"
    assert "boom" in result.error.message


def test_predict_requires_target():
    model, _ = _model_with_fake_pipeline()

    with pytest.raises(InputValidationError, match="target"):
        model.predict_one(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=np.ones((2, 2), dtype=np.uint16),
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0),
            target="",
        )
