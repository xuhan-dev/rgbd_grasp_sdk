from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rgbd_grasp_sdk.types import GraspCandidate, PipelineResult


def pipeline_result_to_dict(result: PipelineResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "best_grasp": _grasp_to_dict(result.best_grasp),
        "candidate_grasps": [_grasp_to_dict(item) for item in result.candidate_grasps],
        "timings": dict(result.timings),
        "metadata": dict(result.metadata),
        "error": asdict(result.error) if result.error is not None else None,
    }


def _grasp_to_dict(grasp: GraspCandidate | None) -> dict[str, Any] | None:
    if grasp is None:
        return None
    return {
        "pose": asdict(grasp.pose),
        "score": grasp.score,
        "center_px": list(grasp.center_px),
        "width": grasp.width,
        "metadata": dict(grasp.metadata),
    }
