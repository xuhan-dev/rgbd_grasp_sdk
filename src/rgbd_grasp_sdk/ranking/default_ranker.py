from __future__ import annotations

import numpy as np

from rgbd_grasp_sdk.types import GraspCandidate


class DefaultGraspRanker:
    def rank(
        self,
        candidates: list[GraspCandidate],
        target_mask: np.ndarray | None = None,
    ) -> list[GraspCandidate]:
        return sorted(candidates, key=lambda item: item.score, reverse=True)
