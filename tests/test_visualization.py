import numpy as np
import pytest

from rgbd_grasp_sdk.types import GraspCandidate, Pose6D
from rgbd_grasp_sdk.visualization.grasp_scene import (
    ALL_GRASP_COLOR,
    SELECTED_GRASP_COLOR,
    _candidate_to_line_set,
)


def test_grasp_visualization_uses_distinct_colors_for_all_and_selected():
    pytest.importorskip("open3d")
    candidate = GraspCandidate(
        pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        score=0.8,
        center_px=(10, 10),
        width=0.06,
    )

    all_grasp = _candidate_to_line_set(candidate, ALL_GRASP_COLOR)
    selected_grasp = _candidate_to_line_set(candidate, SELECTED_GRASP_COLOR)

    all_colors = np.asarray(all_grasp.colors)
    selected_colors = np.asarray(selected_grasp.colors)
    assert np.allclose(all_colors, np.tile(ALL_GRASP_COLOR, (len(all_grasp.lines), 1)))
    assert np.allclose(
        selected_colors,
        np.tile(SELECTED_GRASP_COLOR, (len(selected_grasp.lines), 1)),
    )
    assert not np.allclose(all_colors, selected_colors)
