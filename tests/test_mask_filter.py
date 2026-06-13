import numpy as np

from rgbd_grasp_sdk.filtering.mask_filter import filter_grasps_by_mask
from rgbd_grasp_sdk.masks.postprocess import merge_masks
from rgbd_grasp_sdk.types import GraspCandidate, MaskResult, Pose6D


def _candidate(x: int, y: int, score: float = 0.5) -> GraspCandidate:
    return GraspCandidate(
        pose=Pose6D(x=0.0, y=0.0, z=0.5, roll=0.0, pitch=0.0, yaw=0.0),
        score=score,
        center_px=(x, y),
    )


def test_merge_masks_combines_instances():
    first = np.zeros((4, 4), dtype=bool)
    second = np.zeros((4, 4), dtype=bool)
    first[1, 1] = True
    second[2, 2] = True

    merged = merge_masks([MaskResult(first), MaskResult(second)])

    assert merged[1, 1]
    assert merged[2, 2]
    assert merged.sum() == 2


def test_filter_grasps_by_mask_keeps_only_inside_candidates():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 3] = True
    candidates = [_candidate(3, 2, 0.9), _candidate(0, 0, 0.8), _candidate(99, 99, 1.0)]

    filtered = filter_grasps_by_mask(candidates, mask, min_score=0.0)

    assert len(filtered) == 1
    assert filtered[0].center_px == (3, 2)
