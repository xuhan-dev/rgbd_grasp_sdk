from __future__ import annotations

import rgbd_grasp_sdk.grasping._rng_import as rng_import
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


def test_load_rng_predictor_does_not_install_fallbacks_by_default(tmp_path, monkeypatch):
    _write_fake_rng(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path / "legacy"))

    calls = []
    monkeypatch.setattr(
        rng_import,
        "install_rng_compat_fallbacks",
        lambda: calls.append("called"),
    )
    monkeypatch.setattr(rng_import, "install_mpl_toolkits_namespace_fix", lambda: None)

    load_rng_predictor({"checkpoint_path": "checkpoint.pt"})

    assert calls == []


def test_load_rng_predictor_installs_fallbacks_when_enabled(tmp_path, monkeypatch):
    _write_fake_rng(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path / "legacy"))

    calls = []
    monkeypatch.setattr(
        rng_import,
        "install_rng_compat_fallbacks",
        lambda: calls.append("called"),
    )
    monkeypatch.setattr(rng_import, "install_mpl_toolkits_namespace_fix", lambda: None)

    load_rng_predictor(
        {
            "checkpoint_path": "checkpoint.pt",
            "allow_dependency_fallbacks": True,
        }
    )

    assert calls == ["called"]


def _write_fake_rng(tmp_path):
    legacy_root = tmp_path / "legacy"
    preprocessor_dir = legacy_root / "preprocessor"
    preprocessor_dir.mkdir(parents=True)
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
