from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from rgbd_grasp_sdk.errors import BackendUnavailableError


def load_rng_predictor(options: dict[str, Any]) -> Any:
    GraspPredictor = _load_grasp_predictor_class()

    return GraspPredictor(
        checkpoint_path=str(options["checkpoint_path"]),
        device=str(options.get("device", "cuda")),
        input_size=tuple(options.get("input_size", [360, 640])),
        config=options.get("model_config", {}),
    )


def _load_grasp_predictor_class() -> type[Any]:
    rng_path = _find_rng_module_path()
    module_name = "_rgbd_grasp_sdk_legacy_rng"
    spec = importlib.util.spec_from_file_location(module_name, rng_path)
    if spec is None or spec.loader is None:
        raise BackendUnavailableError(f"无法加载 RNG 模块: {rng_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.GraspPredictor


def _find_rng_module_path() -> Path:
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / "preprocessor" / "RNG.py"
        if candidate.exists():
            return candidate
    raise BackendUnavailableError("无法在 PYTHONPATH 中找到 preprocessor/RNG.py")
