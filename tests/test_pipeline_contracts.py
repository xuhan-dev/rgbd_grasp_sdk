import numpy as np

from rgbd_grasp_sdk.types import (
    CameraIntrinsics,
    GraspCandidate,
    PipelineResult,
    PipelineStatus,
    Pose6D,
)


def test_core_types_are_constructible():
    intrinsics = CameraIntrinsics(fx=600.0, fy=601.0, cx=320.0, cy=240.0)
    pose = Pose6D(x=0.1, y=0.2, z=0.3, roll=0.0, pitch=1.57, yaw=3.14)
    grasp = GraspCandidate(
        pose=pose,
        score=0.9,
        center_px=(100, 120),
        width=0.06,
        metadata={"source": "test"},
    )
    result = PipelineResult(
        status=PipelineStatus.SUCCESS,
        best_grasp=grasp,
        candidate_grasps=[grasp],
        target_mask=np.ones((4, 4), dtype=bool),
        timings={"total": 0.01},
        metadata={"target": "apple"},
    )

    assert intrinsics.matrix.shape == (3, 3)
    assert result.best_grasp is grasp
    assert result.status is PipelineStatus.SUCCESS
