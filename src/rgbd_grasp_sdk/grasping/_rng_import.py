from __future__ import annotations

from typing import Any


def load_rng_predictor(options: dict[str, Any]) -> Any:
    from preprocessor.RNG import GraspPredictor

    return GraspPredictor(
        checkpoint_path=str(options["checkpoint_path"]),
        device=str(options.get("device", "cuda")),
        input_size=tuple(options.get("input_size", [360, 640])),
        config=options.get("model_config", {}),
    )
