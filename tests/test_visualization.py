import numpy as np
import pytest

from rgbd_grasp_sdk.types import GraspCandidate, Pose6D
from rgbd_grasp_sdk.visualization.grasp_scene import (
    ALL_GRASP_COLOR,
    SELECTED_GRASP_COLOR,
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
