from __future__ import annotations

from rgbd_grasp_sdk.types import GraspCandidate


class DefaultGraspRanker:
    def rank(self, candidates: list[GraspCandidate]) -> list[GraspCandidate]:
        return sorted(candidates, key=lambda item: item.score, reverse=True)
