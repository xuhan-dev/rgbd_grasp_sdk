from __future__ import annotations

from rgbd_grasp_sdk.grasping._rng_import import load_rng_predictor


def test_load_rng_predictor_bypasses_legacy_package_init(tmp_path, monkeypatch):
    legacy_root = tmp_path / "legacy"
    preprocessor_dir = legacy_root / "preprocessor"
    preprocessor_dir.mkdir(parents=True)
    (preprocessor_dir / "__init__.py").write_text(
        "raise RuntimeError('package init should not run')\n",
        encoding="utf-8",
    )
    (preprocessor_dir / "RNG.py").write_text(
        "\n".join(
            [
                "class GraspPredictor:",
                "    def __init__(self, checkpoint_path, device, input_size, config):",
                "        self.checkpoint_path = checkpoint_path",
                "        self.device = device",
                "        self.input_size = input_size",
                "        self.config = config",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(legacy_root))

    predictor = load_rng_predictor(
        {
            "checkpoint_path": "checkpoint.pt",
            "device": "cuda",
            "input_size": [360, 640],
            "model_config": {"intrinsics_path": "intrinsics"},
        }
    )

    assert predictor.checkpoint_path == "checkpoint.pt"
    assert predictor.device == "cuda"
    assert predictor.input_size == (360, 640)
    assert predictor.config == {"intrinsics_path": "intrinsics"}
