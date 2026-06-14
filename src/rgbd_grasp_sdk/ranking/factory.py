from __future__ import annotations

from rgbd_grasp_sdk.config.schema import RankingConfig
from rgbd_grasp_sdk.errors import ConfigError
from rgbd_grasp_sdk.ranking.base import GraspRanker
from rgbd_grasp_sdk.ranking.default_ranker import DefaultGraspRanker
from rgbd_grasp_sdk.ranking.mask_aware_ranker import MaskAwareGraspRanker


def create_ranker(config: RankingConfig) -> GraspRanker:
    if config.backend == "default":
        return DefaultGraspRanker(top_k=config.top_k)
    if config.backend == "mask_aware":
        return MaskAwareGraspRanker(config)
    raise ConfigError(f"不支持的 ranking backend: {config.backend}")
