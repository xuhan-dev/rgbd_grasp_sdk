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
        _candidate_to_gripper_mesh(candidate, ALL_GRASP_COLOR) for candidate in all_candidates
    )
    geometries.append(_candidate_to_gripper_mesh(selected_grasp, SELECTED_GRASP_COLOR))
    o3d.visualization.draw_geometries(geometries)


def _candidate_to_gripper_mesh(
    candidate: GraspCandidate,
    color: tuple[float, float, float],
) -> Any:
    import open3d as o3d

    vertices, triangles = _candidate_gripper_mesh(candidate)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (len(vertices), 1)))
    return mesh


def _candidate_gripper_mesh(candidate: GraspCandidate) -> tuple[np.ndarray, np.ndarray]:
    pose = candidate.pose
    rotation = _euler_xyz_to_matrix(pose)
    center = np.array([pose.x, pose.y, pose.z], dtype=float)
    width = float(candidate.width or 0.06)
    scale = 1.0
    finger_width = 0.004 * scale
    height = 0.002 * scale
    tail_length = 0.04
    depth_base = 0.02
    depth = max(width * 0.5, 0.02)

    left_points, left_triangles = _create_mesh_box(
        depth + depth_base + finger_width,
        finger_width,
        height,
    )
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points, right_triangles = _create_mesh_box(
        depth + depth_base + finger_width,
        finger_width,
        height,
    )
    right_triangles += 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points, bottom_triangles = _create_mesh_box(finger_width, width, height)
    bottom_triangles += 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points, tail_triangles = _create_mesh_box(tail_length, finger_width, height)
    tail_triangles += 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate(
        [left_points, right_points, bottom_points, tail_points],
        axis=0,
    )
    vertices = vertices @ rotation.T + center
    triangles = np.concatenate(
        [left_triangles, right_triangles, bottom_triangles, tail_triangles],
        axis=0,
    )
    return vertices, triangles


def _create_mesh_box(width: float, height: float, depth: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0, 0, 0],
            [width, 0, 0],
            [0, 0, depth],
            [width, 0, depth],
            [0, height, 0],
            [width, height, 0],
            [0, height, depth],
            [width, height, depth],
        ],
        dtype=float,
    )
    triangles = np.array(
        [
            [4, 7, 5],
            [4, 6, 7],
            [0, 2, 4],
            [2, 6, 4],
            [0, 1, 2],
            [1, 3, 2],
            [1, 5, 7],
            [1, 7, 3],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 1],
            [1, 4, 5],
        ],
        dtype=np.int32,
    )
    return vertices, triangles


def _euler_xyz_to_matrix(pose: Pose6D) -> np.ndarray:
    roll, pitch, yaw = pose.roll, pose.pitch, pose.yaw
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rx @ ry @ rz
