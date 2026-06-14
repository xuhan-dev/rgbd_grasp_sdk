from __future__ import annotations

from typing import Any

from rgbd_grasp_sdk.types import PipelineResult


def best_grasp_to_transform_message(
    result: PipelineResult,
    *,
    parent_frame: str = "camera_color_optical_frame",
    child_frame: str = "grasp_tcp",
) -> dict[str, Any]:
    if result.best_grasp is None:
        return {
            "parent_frame": parent_frame,
            "child_frame": child_frame,
            "status": result.status.value,
            "error": None if result.error is None else result.error.message,
        }

    grasp = result.best_grasp
    pose = grasp.pose
    return {
        "parent_frame": parent_frame,
        "child_frame": child_frame,
        "translation": {"x": pose.x, "y": pose.y, "z": pose.z},
        "rotation_rpy": {
            "roll": pose.roll,
            "pitch": pose.pitch,
            "yaw": pose.yaw,
        },
        "score": grasp.score,
        "center_px": [grasp.center_px[0], grasp.center_px[1]],
        "width": grasp.width,
    }
