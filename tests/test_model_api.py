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


def test_predict_caches_visualize_override_pipeline_for_batch():
    builder_calls = []
    pipelines = []

    def build_pipeline(config, visualize_3d=None):
        builder_calls.append(visualize_3d)
        pipeline = FakePipeline()
        pipelines.append(pipeline)
        return pipeline

    model = RGBDGrasp(_config(), pipeline_builder=build_pipeline)
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
            "target": "pear",
        },
    ]

    results = model.predict(source=source, visualize_3d=True)

    assert [item.status for item in results] == [
        PipelineStatus.SUCCESS,
        PipelineStatus.SUCCESS,
    ]
    assert builder_calls == [None, True]
    assert len(pipelines[1].calls) == 2


def test_predict_attaches_sample_metadata_to_pipeline_results():
    model, _ = _model_with_fake_pipeline()
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

    assert results[0].metadata["sample_id"] == "a"
    assert results[0].metadata["target"] == "apple"
    assert results[1].metadata["sample_id"] == "b"
    assert results[1].metadata["target"] == "fail"


def test_predict_preserves_existing_pipeline_metadata():
    class MetadataPipeline:
        def run(self, rgb, depth, intrinsics, target):
            return PipelineResult(
                status=PipelineStatus.SUCCESS,
                metadata={
                    "source": "pipeline",
                    "target": "pipeline-target",
                    "sample_id": "pipeline-id",
                },
            )

    model = RGBDGrasp(
        _config(),
        pipeline_builder=lambda config, visualize_3d=None: MetadataPipeline(),
    )

    result = model.predict_one(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0),
        target="apple",
    )

    assert result.metadata["source"] == "pipeline"
    assert result.metadata["target"] == "pipeline-target"
    assert result.metadata["sample_id"] == "pipeline-id"


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


def test_strict_false_converts_input_validation_to_failed_result():
    model, _ = _model_with_fake_pipeline()

    result = model.predict_one(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0),
        target="",
        strict=False,
    )

    assert result.status is PipelineStatus.FAILED
    assert result.error is not None
    assert result.error.code == "input_validation_error"


def test_predict_requires_target():
    model, _ = _model_with_fake_pipeline()

    with pytest.raises(InputValidationError, match="target"):
        model.predict_one(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=np.ones((2, 2), dtype=np.uint16),
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0),
            target="",
        )


def test_info_reports_config_backends_and_dependency_status(tmp_path):
    config = SdkConfig(
        segmentation=SegmentationConfig(
            backend="mock_seg",
            options={"model_path": str(tmp_path / "missing-seg.pt"), "device": "cpu"},
        ),
        grasping=GraspingConfig(
            backend="mock_grasp",
            options={
                "checkpoint_path": str(tmp_path / "missing-rng.ckpt"),
                "device": "cpu",
            },
        ),
        mask=MaskConfig(),
        ranking=RankingConfig(backend="default"),
        outputs=OutputConfig(visualize_3d=False),
    )
    fake = FakePipeline()
    model = RGBDGrasp(
        config,
        pipeline_builder=lambda config, visualize_3d=None: fake,
    )

    info = model.info()

    assert info["segmentation"]["backend"] == "mock_seg"
    assert info["grasping"]["backend"] == "mock_grasp"
    assert info["ranking"]["backend"] == "default"
    assert info["devices"] == {"segmentation": "cpu", "grasping": "cpu"}
    assert "dependencies" in info
    assert info["paths"]["segmentation.model_path"]["exists"] is False
    assert info["paths"]["grasping.checkpoint_path"]["exists"] is False


def test_info_reports_grasping_checkpoint_path_status(tmp_path):
    config = SdkConfig(
        segmentation=SegmentationConfig(
            backend="mock_seg",
            options={"model_path": str(tmp_path / "missing-seg.pt"), "device": "cpu"},
        ),
        grasping=GraspingConfig(
            backend="mock_grasp",
            options={"checkpoint": str(tmp_path / "missing-rng.ckpt"), "device": "cpu"},
        ),
        mask=MaskConfig(),
        ranking=RankingConfig(backend="default"),
        outputs=OutputConfig(visualize_3d=False),
    )
    fake = FakePipeline()
    model = RGBDGrasp(
        config,
        pipeline_builder=lambda config, visualize_3d=None: fake,
    )

    info = model.info()

    assert info["paths"]["grasping.checkpoint"]["exists"] is False


def test_val_converts_malformed_list_items_to_failed_results():
    model, _ = _model_with_fake_pipeline()
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)

    summary = model.val(
        data=[
            {
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth": np.ones((2, 2), dtype=np.uint16),
                "intrinsics": intrinsics,
                "target": "apple",
            },
            {
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "intrinsics": intrinsics,
                "target": "pear",
            },
        ]
    )

    assert summary["total"] == 2
    assert summary["success"] == 1
    assert summary["failed"] == 1


def test_val_uses_manifest_samples_and_validation_summary():
    model, _ = _model_with_fake_pipeline()
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)

    summary = model.val(
        data=[
            {
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth": np.ones((2, 2), dtype=np.uint16),
                "intrinsics": intrinsics,
                "target": "apple",
            },
            {
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth": np.ones((2, 2), dtype=np.uint16),
                "intrinsics": intrinsics,
                "target": "fail",
            },
        ]
    )

    assert summary["total"] == 2
    assert summary["success"] == 1
    assert summary["failed"] == 1


def test_benchmark_converts_malformed_list_items_to_failed_records():
    model, _ = _model_with_fake_pipeline()
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)

    summary = model.benchmark(
        data=[
            {
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth": np.ones((2, 2), dtype=np.uint16),
                "intrinsics": intrinsics,
                "target": "apple",
            },
            {
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "intrinsics": intrinsics,
                "target": "pear",
            },
        ],
        warmup=0,
        repeat=1,
    )

    assert summary["total"] == 2
    assert summary["success"] == 1
    assert summary["failed"] == 1


def test_benchmark_runs_warmup_and_repeat_without_counting_warmup():
    model, fake = _model_with_fake_pipeline()
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    data = [
        {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16),
            "intrinsics": intrinsics,
            "target": "apple",
        }
    ]

    summary = model.benchmark(data=data, warmup=2, repeat=3)

    assert summary["warmup"] == 2
    assert summary["repeat"] == 3
    assert summary["total"] == 3
    assert summary["success"] == 3
    assert len(fake.calls) == 5
    assert summary["backend"]["segmentation"] == "mock_seg"
    assert summary["backend"]["grasping"] == "mock_grasp"


@pytest.mark.parametrize(
    ("warmup", "repeat"),
    [
        (-1, 1),
        (0, 0),
        (0, -1),
    ],
)
def test_benchmark_rejects_invalid_warmup_and_repeat(warmup, repeat):
    model, _ = _model_with_fake_pipeline()
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    data = [
        {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16),
            "intrinsics": intrinsics,
            "target": "apple",
        }
    ]

    with pytest.raises(InputValidationError):
        model.benchmark(data=data, warmup=warmup, repeat=repeat)
