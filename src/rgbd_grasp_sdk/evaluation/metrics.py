from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from rgbd_grasp_sdk.types import PipelineResult, PipelineStatus


def summarize_validation(results: list[PipelineResult]) -> dict[str, Any]:
    total = len(results)
    success_results = [item for item in results if item.status is PipelineStatus.SUCCESS]
    failed_results = [item for item in results if item.status is PipelineStatus.FAILED]

    candidate_counts = [len(item.candidate_grasps) for item in success_results]
    best_scores = [
        float(item.best_grasp.score)
        for item in success_results
        if item.best_grasp is not None
    ]

    return {
        "total": total,
        "success": len(success_results),
        "failed": len(failed_results),
        "success_rate": (len(success_results) / total) if total else 0.0,
        "failure_reasons": _failure_reasons(failed_results),
        "candidate_count": _number_summary(candidate_counts, default_int=True),
        "best_score": _number_summary(best_scores, default_int=False),
        "timings": _timing_means(results),
    }


def _failure_reasons(results: list[PipelineResult]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for result in results:
        if result.error is None:
            counter["unknown"] += 1
        else:
            counter[result.error.code] += 1
    return dict(counter)


def _number_summary(values: list[float | int], *, default_int: bool) -> dict[str, float | int]:
    if not values:
        if default_int:
            return {"mean": 0.0, "min": 0, "max": 0}
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "min": min(values),
        "max": max(values),
    }


def _timing_means(results: list[PipelineResult]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for key, value in result.timings.items():
            buckets[key].append(float(value))
    return {f"{key}_mean": float(mean(values)) for key, values in sorted(buckets.items())}
