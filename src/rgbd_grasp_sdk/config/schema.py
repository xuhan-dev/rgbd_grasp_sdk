from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SegmentationConfig:
    backend: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraspingConfig:
    backend: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaskConfig:
    merge_instances: bool = True
    dilate_kernel: int = 0
    dilate_iterations: int = 0
    min_area: int = 0


@dataclass(frozen=True)
class RankingConfig:
    backend: str = "default"
    top_k: int = 10
    min_target_score: float = 0.0
    require_center_in_mask: bool = False
    weights: dict[str, float] = field(
        default_factory=lambda: {"rng_score": 1.0, "target_score": 0.0}
    )


@dataclass(frozen=True)
class OutputConfig:
    return_point_cloud: bool = False
    return_segmentation_preview: bool = True
    return_candidates: bool = True
    visualize_3d: bool = False


@dataclass(frozen=True)
class SdkConfig:
    segmentation: SegmentationConfig
    grasping: GraspingConfig
    mask: MaskConfig
    ranking: RankingConfig
    outputs: OutputConfig
