from __future__ import annotations

from rgbd_grasp_sdk.types import GraspCandidate, Transform


def grasp_to_transform(
    grasp: GraspCandidate,
    parent_frame: str,
    child_frame: str,
) -> Transform:
    return Transform(parent_frame=parent_frame, child_frame=child_frame, pose=grasp.pose)
