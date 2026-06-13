import importlib.util
from pathlib import Path


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
