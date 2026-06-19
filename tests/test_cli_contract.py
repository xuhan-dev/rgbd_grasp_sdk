import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_cli_module():
    module_path = REPO_ROOT / "examples" / "run_image_pair.py"
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


def test_main_uses_rgbd_grasp_predict_one(monkeypatch, tmp_path, capsys):
    from examples import run_image_pair
    from rgbd_grasp_sdk.types import GraspCandidate, PipelineResult, PipelineStatus, Pose6D

    calls = {}

    class FakeModel:
        def __init__(self, config):
            calls["config"] = config

        def predict_one(
            self,
            *,
            rgb,
            depth,
            intrinsics,
            target,
            visualize_3d=None,
            output_json=None,
            output_transform_json=None,
        ):
            calls["predict_one"] = {
                "rgb": rgb,
                "depth": depth,
                "intrinsics": intrinsics,
                "target": target,
                "visualize_3d": visualize_3d,
                "output_json": output_json,
                "output_transform_json": output_transform_json,
            }
            grasp = GraspCandidate(
                pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
                score=0.9,
                center_px=(1, 2),
            )
            return PipelineResult(
                status=PipelineStatus.SUCCESS,
                best_grasp=grasp,
                timings={"total": 0.1},
            )

    monkeypatch.setattr(run_image_pair, "RGBDGrasp", FakeModel)
    run_image_pair.main(
        [
            "--config",
            "config.yaml",
            "--rgb",
            "rgb.png",
            "--depth",
            "depth.png",
            "--intrinsics",
            "K.npz",
            "--target",
            "apple",
            "--output-json",
            str(tmp_path / "result.json"),
            "--output-transform-json",
            str(tmp_path / "tf.json"),
            "--no-visualize-3d",
        ]
    )

    assert calls["config"] == "config.yaml"
    assert calls["predict_one"]["rgb"] == "rgb.png"
    assert calls["predict_one"]["depth"] == "depth.png"
    assert calls["predict_one"]["intrinsics"] == "K.npz"
    assert calls["predict_one"]["target"] == "apple"
    assert calls["predict_one"]["visualize_3d"] is False
    assert calls["predict_one"]["output_json"] == str(tmp_path / "result.json")
    assert calls["predict_one"]["output_transform_json"] == str(tmp_path / "tf.json")
    assert "status: success" in capsys.readouterr().out
