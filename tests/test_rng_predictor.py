import numpy as np
import pytest

from rgbd_grasp_sdk.errors import BackendUnavailableError
from rgbd_grasp_sdk.grasping.rng_adapter_utils import rng_grasp_to_candidate
from rgbd_grasp_sdk.grasping.rng_predictor import RngGraspPredictor
from rgbd_grasp_sdk.types import CameraIntrinsics, GraspRequest


class FakeRngGrasp:
    score = 0.7
    width = 0.05
    height = 0.002
    depth = 0.03
    translation = np.array([0.1, 0.2, 0.3])
    rotation = np.eye(3)


def test_rng_grasp_to_candidate_converts_pose_and_center():
    candidate = rng_grasp_to_candidate(FakeRngGrasp(), center_px=(12, 34))

    assert candidate.score == 0.7
    assert candidate.width == 0.05
    assert candidate.center_px == (12, 34)
    assert candidate.pose.x == 0.1
    assert candidate.pose.z == 0.3
    assert candidate.metadata["rotation_matrix"] == np.eye(3).tolist()
    assert candidate.metadata["depth"] == 0.03


def test_rng_predictor_reports_missing_checkpoint_before_importing_heavy_model():
    with pytest.raises(BackendUnavailableError, match="checkpoint_path"):
        RngGraspPredictor({})


def test_rng_predictor_projects_translation_to_center_px_when_missing():
    class FakePredictor:
        def predict(self, rgb, depth, vis=False):
            return [FakeRngGrasp()], None

    predictor = RngGraspPredictor({}, predictor=FakePredictor())
    result = predictor.predict(
        GraspRequest(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=np.zeros((2, 2), dtype=np.uint16),
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=10.0, cy=20.0),
        )
    )

    assert result.candidates[0].center_px == (43, 87)


def test_rng_predictor_does_not_visualize_inside_backend():
    class FakePredictor:
        def __init__(self):
            self.vis_values = []

        def predict(self, rgb, depth, vis=False):
            self.vis_values.append(vis)
            return [FakeRngGrasp()], None

    backend = FakePredictor()
    predictor = RngGraspPredictor({}, predictor=backend)
    predictor.predict(
        GraspRequest(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=np.zeros((2, 2), dtype=np.uint16),
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=10.0, cy=20.0),
            metadata={"visualize_3d": True},
        )
    )

    assert backend.vis_values == [False]
