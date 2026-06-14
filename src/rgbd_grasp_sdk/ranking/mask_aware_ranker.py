from __future__ import annotations

import numpy as np

from rgbd_grasp_sdk.config.schema import RankingConfig
from rgbd_grasp_sdk.types import GraspCandidate


class MaskAwareGraspRanker:
    def __init__(self, config: RankingConfig):
        self.config = config

    def rank(
        self,
        candidates: list[GraspCandidate],
        target_mask: np.ndarray | None = None,
    ) -> list[GraspCandidate]:
        scored = [self._with_scores(candidate, target_mask) for candidate in candidates]
        filtered = [
            candidate
            for candidate in scored
            if candidate.metadata["target_score"] >= self.config.min_target_score
        ]
        if self.config.require_center_in_mask:
            filtered = [
                candidate
                for candidate in filtered
                if candidate.metadata["center_in_mask"]
            ]
        return sorted(
            filtered,
            key=lambda candidate: candidate.metadata["final_score"],
            reverse=True,
        )[: self.config.top_k]

    def _with_scores(
        self,
        candidate: GraspCandidate,
        target_mask: np.ndarray | None,
    ) -> GraspCandidate:
        center_in_mask = _center_in_mask(candidate, target_mask)
        overlap_ratio = _mask_overlap_ratio(candidate, target_mask)
        target_score = max(1.0 if center_in_mask else 0.0, overlap_ratio)
        rng_weight = float(self.config.weights.get("rng_score", 1.0))
        target_weight = float(self.config.weights.get("target_score", 0.0))
        final_score = candidate.score * rng_weight + target_score * target_weight
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "center_in_mask": center_in_mask,
                "mask_overlap_ratio": float(overlap_ratio),
                "target_score": float(target_score),
                "final_score": float(final_score),
            }
        )
        return GraspCandidate(
            pose=candidate.pose,
            score=candidate.score,
            center_px=candidate.center_px,
            width=candidate.width,
            metadata=metadata,
        )


def _center_in_mask(
    candidate: GraspCandidate,
    target_mask: np.ndarray | None,
) -> bool:
    if target_mask is None or target_mask.size == 0:
        return False
    x, y = candidate.center_px
    if y < 0 or x < 0 or y >= target_mask.shape[0] or x >= target_mask.shape[1]:
        return False
    return bool(target_mask[y, x])


def _mask_overlap_ratio(
    candidate: GraspCandidate,
    target_mask: np.ndarray | None,
) -> float:
    if target_mask is None or target_mask.size == 0:
        return 0.0
    x, y = candidate.center_px
    radius = _candidate_pixel_radius(candidate)
    x_min = max(0, x - radius)
    x_max = min(target_mask.shape[1], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(target_mask.shape[0], y + radius + 1)
    if x_min >= x_max or y_min >= y_max:
        return 0.0
    target_area = int(np.count_nonzero(target_mask))
    if target_area == 0:
        return 0.0
    region = target_mask[y_min:y_max, x_min:x_max]
    return float(np.count_nonzero(region) / target_area)


def _candidate_pixel_radius(candidate: GraspCandidate) -> int:
    if candidate.width is None:
        return 3
    return max(3, int(round(float(candidate.width) * 100.0)))
