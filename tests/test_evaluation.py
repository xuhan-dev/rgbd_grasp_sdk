from __future__ import annotations

import pytest

from rgbd_grasp_sdk.evaluation import summarize_validation
from rgbd_grasp_sdk.types import (
    GraspCandidate,
    PipelineError,
    PipelineResult,
    PipelineStatus,
    Pose6D,
)


def _grasp(score: float) -> GraspCandidate:
    return GraspCandidate(
        pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        score=score,
        center_px=(10, 20),
    )


def test_summarize_validation_counts_success_failure_and_reasons():
    results = [
        PipelineResult(
            status=PipelineStatus.SUCCESS,
            best_grasp=_grasp(0.8),
            candidate_grasps=[_grasp(0.8), _grasp(0.7)],
            timings={"total": 0.5, "segmentation": 0.1, "grasping": 0.3},
        ),
        PipelineResult(
            status=PipelineStatus.FAILED,
            error=PipelineError(code="empty_mask", message="目标 mask 为空"),
            timings={"total": 0.2, "segmentation": 0.2},
        ),
        PipelineResult(
            status=PipelineStatus.FAILED,
            error=PipelineError(code="empty_mask", message="目标 mask 为空"),
            timings={"total": 0.4},
        ),
    ]

    summary = summarize_validation(results)

    assert summary["total"] == 3
    assert summary["success"] == 1
    assert summary["failed"] == 2
    assert summary["success_rate"] == 1 / 3
    assert summary["failure_reasons"] == {"empty_mask": 2}
    assert summary["candidate_count"] == {"mean": 2.0, "min": 2, "max": 2}
    assert summary["best_score"] == {"mean": 0.8, "min": 0.8, "max": 0.8}
    assert summary["timings"]["total_mean"] == pytest.approx(0.3666666666666667)
    assert summary["timings"]["segmentation_mean"] == pytest.approx(0.15000000000000002)
    assert summary["timings"]["grasping_mean"] == pytest.approx(0.3)


def test_summarize_validation_handles_empty_results():
    summary = summarize_validation([])

    assert summary == {
        "total": 0,
        "success": 0,
        "failed": 0,
        "success_rate": 0.0,
        "failure_reasons": {},
        "candidate_count": {"mean": 0.0, "min": 0, "max": 0},
        "best_score": {"mean": 0.0, "min": 0.0, "max": 0.0},
        "timings": {},
    }
