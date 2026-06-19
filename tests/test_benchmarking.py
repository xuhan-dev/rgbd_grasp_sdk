from __future__ import annotations

import pytest

from rgbd_grasp_sdk.benchmarking import BenchmarkRecord, summarize_benchmark
from rgbd_grasp_sdk.types import PipelineError, PipelineResult, PipelineStatus


def _result(status: PipelineStatus, **timings):
    return PipelineResult(status=status, timings=dict(timings))


def test_summarize_benchmark_reports_percentiles_throughput_and_failures():
    records = [
        BenchmarkRecord(result=_result(PipelineStatus.SUCCESS, total=0.10, grasping=0.07), elapsed=0.11),
        BenchmarkRecord(result=_result(PipelineStatus.SUCCESS, total=0.20, grasping=0.12), elapsed=0.21),
        BenchmarkRecord(
            result=PipelineResult(
                status=PipelineStatus.FAILED,
                error=PipelineError(code="empty_mask", message="目标 mask 为空"),
                timings={"total": 0.30},
            ),
            elapsed=0.31,
        ),
    ]

    summary = summarize_benchmark(
        records,
        warmup=1,
        repeat=2,
        backend_summary={"segmentation": "yolo", "grasping": "rng"},
    )

    assert summary["warmup"] == 1
    assert summary["repeat"] == 2
    assert summary["total"] == 3
    assert summary["success"] == 2
    assert summary["failed"] == 1
    assert summary["failure_reasons"] == {"empty_mask": 1}
    assert summary["elapsed"]["mean"] == pytest.approx(0.21)
    assert summary["elapsed"]["p50"] == 0.21
    assert summary["elapsed"]["p95"] == 0.31
    assert summary["elapsed"]["max"] == 0.31
    assert summary["timings"]["total"]["mean"] == pytest.approx(0.2)
    assert summary["timings"]["grasping"]["mean"] == pytest.approx(0.095)
    assert summary["samples_per_sec"] == pytest.approx(1 / 0.21)
    assert summary["backend"] == {"segmentation": "yolo", "grasping": "rng"}


def test_summarize_benchmark_handles_no_records():
    summary = summarize_benchmark([], warmup=0, repeat=1, backend_summary={})

    assert summary["total"] == 0
    assert summary["elapsed"]["mean"] == pytest.approx(0.0)
    assert summary["elapsed"]["p50"] == 0.0
    assert summary["elapsed"]["p95"] == 0.0
    assert summary["elapsed"]["max"] == 0.0
    assert summary["samples_per_sec"] == pytest.approx(0.0)
