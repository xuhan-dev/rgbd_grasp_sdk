from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import fsum
from typing import Any

from rgbd_grasp_sdk.types import PipelineResult, PipelineStatus


@dataclass(frozen=True)
class BenchmarkRecord:
    result: PipelineResult
    elapsed: float


def summarize_benchmark(
    records: list[BenchmarkRecord],
    *,
    warmup: int,
    repeat: int,
    backend_summary: dict[str, Any],
) -> dict[str, Any]:
    total = len(records)
    success = sum(1 for record in records if record.result.status is PipelineStatus.SUCCESS)
    failed = total - success
    elapsed_values = [float(record.elapsed) for record in records]

    elapsed = _distribution(elapsed_values)
    return {
        "warmup": warmup,
        "repeat": repeat,
        "total": total,
        "success": success,
        "failed": failed,
        "success_rate": (success / total) if total else 0.0,
        "failure_reasons": _failure_reasons(records),
        "elapsed": elapsed,
        "timings": _timing_distributions(records),
        "samples_per_sec": (1.0 / elapsed["mean"]) if elapsed["mean"] > 0 else 0.0,
        "backend": dict(backend_summary),
    }


def _failure_reasons(records: list[BenchmarkRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        if record.result.status is PipelineStatus.SUCCESS:
            continue
        if record.result.error is None:
            counter["unknown"] += 1
        else:
            counter[record.result.error.code] += 1
    return dict(counter)


def _timing_distributions(records: list[BenchmarkRecord]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, value in record.result.timings.items():
            buckets[key].append(float(value))
    return {key: _distribution(values) for key, values in sorted(buckets.items())}


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "mean": fsum(ordered) / len(ordered),
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "max": ordered[-1],
    }


def _nearest_rank(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]
