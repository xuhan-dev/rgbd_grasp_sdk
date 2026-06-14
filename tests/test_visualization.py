import numpy as np
import pytest

from rgbd_grasp_sdk.types import GraspCandidate, Pose6D
from rgbd_grasp_sdk.visualization.grasp_scene import (
    ALL_GRASP_COLOR,
    SELECTED_GRASP_COLOR,
    TARGET_MASK_OVERLAY_COLOR,
    _apply_target_mask_overlay,
    _candidate_gripper_mesh,
    _candidate_to_gripper_mesh,
)


def test_grasp_visualization_uses_distinct_colors_for_all_and_selected():
    pytest.importorskip("open3d")
    candidate = GraspCandidate(
        pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        score=0.8,
        center_px=(10, 10),
        width=0.06,
    )

    all_grasp = _candidate_to_gripper_mesh(candidate, ALL_GRASP_COLOR)
    selected_grasp = _candidate_to_gripper_mesh(candidate, SELECTED_GRASP_COLOR)

    all_colors = np.asarray(all_grasp.vertex_colors)
    selected_colors = np.asarray(selected_grasp.vertex_colors)
    assert len(all_grasp.triangles) == 48
    assert len(all_grasp.vertices) == 32
    assert np.allclose(
        all_colors,
        np.tile(ALL_GRASP_COLOR, (len(all_grasp.vertices), 1)),
    )
    assert np.allclose(
        selected_colors,
        np.tile(SELECTED_GRASP_COLOR, (len(selected_grasp.vertices), 1)),
    )
    assert not np.allclose(all_colors, selected_colors)


def test_target_mask_overlay_preserves_base_point_cloud_colors():
    o3d = pytest.importorskip("open3d")
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
    )
    base_colors = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6],
        ],
        dtype=float,
    )
    point_cloud.colors = o3d.utility.Vector3dVector(base_colors)
    target_mask = np.array([[True, False], [False, True]])

    colored = _apply_target_mask_overlay(point_cloud, target_mask)

    colors = np.asarray(colored.colors)
    expected_target_color = base_colors[0] * 0.45 + np.asarray(TARGET_MASK_OVERLAY_COLOR) * 0.55
    assert np.allclose(colors[0], expected_target_color)
    assert np.allclose(colors[1], base_colors[1])
    assert np.allclose(colors[2], base_colors[2])
    assert not np.shares_memory(colors, base_colors)


def test_gripper_mesh_uses_raw_rng_rotation_and_depth_metadata():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    candidate = GraspCandidate(
        pose=Pose6D(0.1, 0.2, 0.3, 0.0, 0.0, 0.0),
        score=0.8,
        center_px=(10, 10),
        width=0.06,
        metadata={
            "rotation_matrix": rotation.tolist(),
            "depth": 0.03,
        },
    )

    vertices, _ = _candidate_gripper_mesh(candidate)

    center = np.array([candidate.pose.x, candidate.pose.y, candidate.pose.z])
    local = (vertices - center) @ rotation
    assert np.isclose(local[:, 0].min(), -0.064)
    assert np.isclose(local[:, 0].max(), 0.03)
    assert np.isclose(local[:, 1].min(), -0.034)
    assert np.isclose(local[:, 1].max(), 0.034)
