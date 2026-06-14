import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from rgbd_grasp_sdk.types import GraspCandidate, PipelineResult, PipelineStatus, Pose6D


def _load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "run_image_pair.py"
    spec = importlib.util.spec_from_file_location("run_image_pair", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_supports_output_json():
    cli = _load_cli_module()
    args = cli.parse_args(
        [
            "--config",
            "configs/yolo_rng.yaml",
            "--rgb",
            "data/rgb.png",
            "--depth",
            "data/depth.png",
            "--intrinsics",
            "data/camera_intrinsics.npz",
            "--target",
            "apple",
            "--output-json",
            "outputs/result.json",
        ]
    )

    assert args.output_json == "outputs/result.json"


def test_parse_args_supports_output_transform_json():
    cli = _load_cli_module()
    args = cli.parse_args(
        [
            "--config",
            "configs/yolo_rng.yaml",
            "--rgb",
            "data/rgb.png",
            "--depth",
            "data/depth.png",
            "--intrinsics",
            "data/camera_intrinsics.npz",
            "--target",
            "apple",
            "--output-transform-json",
            "outputs/grasp_tf.json",
        ]
    )

    assert args.output_transform_json == "outputs/grasp_tf.json"


def test_publish_outputs_writes_result_and_transform_json(tmp_path):
    cli = _load_cli_module()
    result_path = tmp_path / "result.json"
    transform_path = tmp_path / "grasp_tf.json"
    result = PipelineResult(
        status=PipelineStatus.SUCCESS,
        best_grasp=GraspCandidate(
            pose=Pose6D(x=0.1, y=0.2, z=0.3, roll=0.0, pitch=0.1, yaw=0.2),
            score=0.9,
            center_px=(320, 240),
            width=0.06,
        ),
    )

    cli._publish_outputs(
        result,
        SimpleNamespace(
            output_json=str(result_path),
            output_transform_json=str(transform_path),
        ),
    )

    result_data = json.loads(result_path.read_text(encoding="utf-8"))
    transform_data = json.loads(transform_path.read_text(encoding="utf-8"))
    assert result_data["best_grasp"]["score"] == 0.9
    assert transform_data["child_frame"] == "grasp_tcp"
    assert transform_data["translation"] == {"x": 0.1, "y": 0.2, "z": 0.3}


def test_parse_args_supports_visualize_3d_override():
    cli = _load_cli_module()
    enabled = cli.parse_args(
        [
            "--config",
            "configs/yolo_rng.yaml",
            "--rgb",
            "data/rgb.png",
            "--depth",
            "data/depth.png",
            "--intrinsics",
            "data/camera_intrinsics.npz",
            "--target",
            "apple",
            "--visualize-3d",
        ]
    )
    disabled = cli.parse_args(
        [
            "--config",
            "configs/yolo_rng.yaml",
            "--rgb",
            "data/rgb.png",
            "--depth",
            "data/depth.png",
            "--intrinsics",
            "data/camera_intrinsics.npz",
            "--target",
            "apple",
            "--no-visualize-3d",
        ]
    )

    assert enabled.visualize_3d is True
    assert disabled.visualize_3d is False
