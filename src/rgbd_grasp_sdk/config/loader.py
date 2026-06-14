from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rgbd_grasp_sdk.config.schema import (
    GraspingConfig,
    MaskConfig,
    OutputConfig,
    RankingConfig,
    SdkConfig,
    SegmentationConfig,
)
from rgbd_grasp_sdk.errors import ConfigError


REQUIRED_SECTIONS = ("segmentation", "grasping", "mask", "ranking", "outputs")


def load_config(path: str | Path) -> SdkConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"配置文件不存在: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("配置文件根节点必须是对象")

    for section in REQUIRED_SECTIONS:
        if section not in raw:
            raise ConfigError(f"缺少配置段: {section}")

    return SdkConfig(
        segmentation=_load_segmentation(raw["segmentation"]),
        grasping=_load_grasping(raw["grasping"]),
        mask=_load_mask(raw["mask"]),
        ranking=_load_ranking(raw["ranking"]),
        outputs=_load_outputs(raw["outputs"]),
    )


def _require_backend(section_name: str, data: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(data, dict):
        raise ConfigError(f"{section_name} 配置段必须是对象")
    backend = data.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ConfigError(f"{section_name}.backend 必须是非空字符串")
    options = {key: value for key, value in data.items() if key != "backend"}
    return backend, options


def _load_segmentation(data: Any) -> SegmentationConfig:
    backend, options = _require_backend("segmentation", data)
    return SegmentationConfig(backend=backend, options=options)


def _load_grasping(data: Any) -> GraspingConfig:
    backend, options = _require_backend("grasping", data)
    return GraspingConfig(backend=backend, options=options)


def _load_mask(data: Any) -> MaskConfig:
    if not isinstance(data, dict):
        raise ConfigError("mask 配置段必须是对象")
    return MaskConfig(
        merge_instances=bool(data.get("merge_instances", True)),
        dilate_kernel=int(data.get("dilate_kernel", 0)),
        dilate_iterations=int(data.get("dilate_iterations", 0)),
        min_area=int(data.get("min_area", 0)),
    )


def _load_ranking(data: Any) -> RankingConfig:
    backend, options = _require_backend("ranking", data)
    raw_weights = options.get("weights", {})
    if raw_weights is None:
        raw_weights = {}
    if not isinstance(raw_weights, dict):
        raise ConfigError("ranking.weights 必须是对象")
    weights = {
        "rng_score": float(raw_weights.get("rng_score", 1.0)),
        "target_score": float(raw_weights.get("target_score", 0.0)),
    }
    return RankingConfig(
        backend=backend,
        top_k=int(options.get("top_k", 10)),
        min_target_score=float(options.get("min_target_score", 0.0)),
        require_center_in_mask=bool(options.get("require_center_in_mask", False)),
        weights=weights,
    )


def _load_outputs(data: Any) -> OutputConfig:
    if not isinstance(data, dict):
        raise ConfigError("outputs 配置段必须是对象")
    return OutputConfig(
        return_point_cloud=bool(data.get("return_point_cloud", False)),
        return_segmentation_preview=bool(data.get("return_segmentation_preview", True)),
        return_candidates=bool(data.get("return_candidates", True)),
        visualize_3d=bool(data.get("visualize_3d", False)),
    )
