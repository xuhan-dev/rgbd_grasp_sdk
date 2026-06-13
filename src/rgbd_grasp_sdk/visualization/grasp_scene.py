from __future__ import annotations

from math import cos, sin
from typing import Any

import numpy as np

from rgbd_grasp_sdk.types import GraspCandidate, Pose6D


ALL_GRASP_COLOR = (0.55, 0.55, 0.55)
SELECTED_GRASP_COLOR = (1.0, 0.0, 0.0)


def visualize_grasp_candidates(
    point_cloud: Any,
    all_candidates: list[GraspCandidate],
    selected_grasp: GraspCandidate,
) -> None:
    import open3d as o3d

    geometries = []
    if point_cloud is not None:
        geometries.append(point_cloud)

    geometries.extend(
        _candidate_to_line_set(candidate, ALL_GRASP_COLOR) for candidate in all_candidates
    )
    geometries.append(_candidate_to_line_set(selected_grasp, SELECTED_GRASP_COLOR))
    o3d.visualization.draw_geometries(geometries)


def _candidate_to_line_set(
    candidate: GraspCandidate,
    color: tuple[float, float, float],
) -> Any:
    import open3d as o3d

    points, lines = _candidate_gripper_lines(candidate)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return line_set


def _candidate_gripper_lines(candidate: GraspCandidate) -> tuple[np.ndarray, np.ndarray]:
    pose = candidate.pose
    rotation = _euler_xyz_to_matrix(pose)
    center = np.array([pose.x, pose.y, pose.z], dtype=float)
    width = float(candidate.width or 0.06)
    finger_length = max(width * 0.7, 0.03)
    tail_length = max(width * 0.5, 0.025)

    local_points = np.array(
        [
            [0.0, -width / 2.0, 0.0],
            [0.0, width / 2.0, 0.0],
            [finger_length, -width / 2.0, 0.0],
            [finger_length, width / 2.0, 0.0],
            [-tail_length, 0.0, 0.0],
        ],
        dtype=float,
    )
    points = local_points @ rotation.T + center
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [1, 3],
            [4, 0],
            [4, 1],
        ],
        dtype=np.int32,
    )
    return points, lines


def _euler_xyz_to_matrix(pose: Pose6D) -> np.ndarray:
    roll, pitch, yaw = pose.roll, pose.pitch, pose.yaw
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rx @ ry @ rz
