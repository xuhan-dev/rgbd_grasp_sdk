from __future__ import annotations

from typing import Protocol

import numpy as np

from rgbd_grasp_sdk.types import GraspCandidate


class GraspRanker(Protocol):
    def rank(
        self,
        candidates: list[GraspCandidate],
        target_mask: np.ndarray | None = None,
    ) -> list[GraspCandidate]:
        """返回按优先级排序后的抓取候选。"""
