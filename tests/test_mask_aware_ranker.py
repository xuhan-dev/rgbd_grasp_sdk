import numpy as np

from rgbd_grasp_sdk.config.schema import RankingConfig
from rgbd_grasp_sdk.ranking.mask_aware_ranker import MaskAwareGraspRanker
from rgbd_grasp_sdk.types import GraspCandidate, Pose6D


def _candidate(score, center_px):
    return GraspCandidate(
        pose=Pose6D(x=0.0, y=0.0, z=0.5, roll=0.0, pitch=0.0, yaw=0.0),
        score=score,
        center_px=center_px,
        width=0.06,
    )


def test_mask_aware_ranker_prefers_candidate_inside_target_mask():
    mask = np.zeros((40, 40), dtype=bool)
    mask[15:25, 15:25] = True
    outside_high_score = _candidate(0.95, (0, 0))
    inside_lower_score = _candidate(0.70, (20, 20))
    ranker = MaskAwareGraspRanker(
        RankingConfig(
            backend="mask_aware",
            weights={"rng_score": 0.5, "target_score": 0.5},
        )
    )

    ranked = ranker.rank([outside_high_score, inside_lower_score], target_mask=mask)

    assert ranked[0].center_px == inside_lower_score.center_px
    assert ranked[0].metadata["target_score"] == 1.0
    assert ranked[1].metadata["target_score"] == 0.0


def test_mask_aware_ranker_can_require_center_inside_mask():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    ranker = MaskAwareGraspRanker(
        RankingConfig(
            backend="mask_aware",
            require_center_in_mask=True,
        )
    )

    ranked = ranker.rank(
        [_candidate(0.95, (1, 1)), _candidate(0.70, (10, 10))],
        target_mask=mask,
    )

    assert len(ranked) == 1
    assert ranked[0].center_px == (10, 10)


def test_mask_aware_ranker_scores_gripper_overlap_with_target_mask():
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:15, 14:18] = True
    overlap_candidate = _candidate(0.80, (10, 12))
    far_candidate = _candidate(0.95, (1, 1))
    ranker = MaskAwareGraspRanker(
        RankingConfig(
            backend="mask_aware",
            weights={"rng_score": 0.4, "target_score": 0.6},
        )
    )

    ranked = ranker.rank([far_candidate, overlap_candidate], target_mask=mask)

    assert ranked[0].center_px == overlap_candidate.center_px
    assert ranked[0].metadata["center_in_mask"] is False
    assert ranked[0].metadata["mask_overlap_ratio"] > 0.0
    assert ranked[0].metadata["target_score"] == ranked[0].metadata["mask_overlap_ratio"]
    assert ranked[1].metadata["mask_overlap_ratio"] == 0.0
