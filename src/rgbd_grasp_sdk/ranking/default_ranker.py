from __future__ import annotations

import numpy as np

from rgbd_grasp_sdk.types import GraspCandidate


class DefaultGraspRanker:
    def __init__(self, top_k: int | None = None):
        self.top_k = top_k

    def rank(
        self,
        candidates: list[GraspCandidate],
        target_mask: np.ndarray | None = None,
    ) -> list[GraspCandidate]:
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        if self.top_k is None:
            return ranked
        return ranked[: self.top_k]
