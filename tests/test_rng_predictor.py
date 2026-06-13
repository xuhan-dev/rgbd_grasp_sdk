import numpy as np

from rgbd_grasp_sdk.grasping.rng_adapter_utils import rng_grasp_to_candidate


class FakeRngGrasp:
    score = 0.7
    width = 0.05
    translation = np.array([0.1, 0.2, 0.3])
    rotation = np.eye(3)


def test_rng_grasp_to_candidate_converts_pose_and_center():
    candidate = rng_grasp_to_candidate(FakeRngGrasp(), center_px=(12, 34))

    assert candidate.score == 0.7
    assert candidate.width == 0.05
    assert candidate.center_px == (12, 34)
    assert candidate.pose.x == 0.1
    assert candidate.pose.z == 0.3
