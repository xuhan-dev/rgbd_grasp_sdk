# Ultralytics-Style Python API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-level `RGBDGrasp` Python API for inference, unlabeled validation, and benchmarking while preserving the current adapter/factory/pipeline architecture.

**Architecture:** `RGBDGrasp` is a thin orchestration object over the existing config loader, backend factories, IO helpers, publishers, and `GraspPipeline`. Input normalization and manifest parsing live in `datasets/`; validation and benchmark aggregation live in focused pure-function modules so they can be tested without real model weights or GPU dependencies.

**Tech Stack:** Python 3.10, dataclasses, pathlib, json, time.perf_counter, statistics, importlib, NumPy, PyYAML, pytest, existing `rgbd_grasp_sdk` types, pipeline, IO, config, publishers.

---

## File Structure

- Create: `src/rgbd_grasp_sdk/datasets/__init__.py`
  - Export manifest/sample helpers.
- Create: `src/rgbd_grasp_sdk/datasets/manifest.py`
  - Define `GraspSample`, normalize single/batch sources, load YAML/JSON manifests, resolve relative paths.
- Create: `src/rgbd_grasp_sdk/evaluation/__init__.py`
  - Export validation aggregation.
- Create: `src/rgbd_grasp_sdk/evaluation/metrics.py`
  - Aggregate `PipelineResult` objects into unlabeled engineering metrics.
- Create: `src/rgbd_grasp_sdk/benchmarking/__init__.py`
  - Export benchmark aggregation helpers.
- Create: `src/rgbd_grasp_sdk/benchmarking/stats.py`
  - Aggregate repeated benchmark records into timing percentiles, throughput, backend summary, and failure metrics.
- Create: `src/rgbd_grasp_sdk/model.py`
  - Implement `RGBDGrasp`, input loading, strict/non-strict error conversion, `predict_one()`, `predict()`, `info()`, `val()`, `benchmark()`.
- Modify: `src/rgbd_grasp_sdk/__init__.py`
  - Export `RGBDGrasp`.
- Modify: `examples/run_image_pair.py`
  - In the final task only, reuse `RGBDGrasp.predict_one()` while keeping current CLI flags and output behavior.
- Test: `tests/test_manifest_dataset.py`
  - YAML/JSON manifest parsing, relative paths, single-frame normalization, missing fields.
- Test: `tests/test_evaluation.py`
  - Success/failure counts, failure reasons, candidate count, best score, timing means.
- Test: `tests/test_benchmarking.py`
  - Warmup/repeat aggregation, percentiles, throughput, failure rate.
- Test: `tests/test_model_api.py`
  - Top-level API export, initialization with injected pipeline builder, array/path input, strict behavior, predict list behavior, info, val, benchmark.
- Test: `tests/test_cli_contract.py`
  - Existing CLI behavior remains intact after it reuses `RGBDGrasp`.

---

### Task 1: Manifest And Sample Normalization

**Files:**
- Create: `src/rgbd_grasp_sdk/datasets/__init__.py`
- Create: `src/rgbd_grasp_sdk/datasets/manifest.py`
- Test: `tests/test_manifest_dataset.py`

- [ ] **Step 1: Write failing manifest tests**

Create `tests/test_manifest_dataset.py`:

```python
from __future__ import annotations

import json

import numpy as np
import pytest

from rgbd_grasp_sdk.datasets import GraspSample, load_samples, normalize_samples
from rgbd_grasp_sdk.errors import InputValidationError
from rgbd_grasp_sdk.types import CameraIntrinsics


def test_normalize_single_keyword_sample_preserves_arrays():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.ones((2, 3), dtype=np.uint16)
    intrinsics = CameraIntrinsics(fx=1.0, fy=2.0, cx=3.0, cy=4.0)

    samples = normalize_samples(
        source=None,
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        target="apple",
    )

    assert samples == [
        GraspSample(
            id=None,
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            target="apple",
        )
    ]


def test_normalize_source_list_keeps_sample_ids():
    samples = normalize_samples(
        source=[
            {
                "id": "a",
                "rgb": "a/rgb.png",
                "depth": "a/depth.png",
                "intrinsics": "a/K.npz",
                "target": "apple",
            }
        ]
    )

    assert len(samples) == 1
    assert samples[0].id == "a"
    assert str(samples[0].rgb) == "a/rgb.png"
    assert str(samples[0].depth) == "a/depth.png"
    assert str(samples[0].intrinsics) == "a/K.npz"
    assert samples[0].target == "apple"


def test_load_yaml_manifest_resolves_relative_paths(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
- id: sample-a
  rgb: images/rgb.png
  depth: images/depth.png
  intrinsics: camera/K.npz
  target: apple
""",
        encoding="utf-8",
    )

    samples = load_samples(manifest)

    assert samples[0].id == "sample-a"
    assert samples[0].rgb == tmp_path / "images/rgb.png"
    assert samples[0].depth == tmp_path / "images/depth.png"
    assert samples[0].intrinsics == tmp_path / "camera/K.npz"
    assert samples[0].target == "apple"


def test_load_json_manifest_resolves_relative_paths(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "rgb": "rgb.png",
                    "depth": "depth.png",
                    "intrinsics": "K.npz",
                    "target": "banana",
                }
            ]
        ),
        encoding="utf-8",
    )

    samples = load_samples(manifest)

    assert samples[0].rgb == tmp_path / "rgb.png"
    assert samples[0].depth == tmp_path / "depth.png"
    assert samples[0].intrinsics == tmp_path / "K.npz"
    assert samples[0].target == "banana"


def test_missing_required_sample_field_raises_clear_error():
    with pytest.raises(InputValidationError, match="sample\\[0\\].depth"):
        normalize_samples(
            source=[
                {
                    "rgb": "rgb.png",
                    "intrinsics": "K.npz",
                    "target": "apple",
                }
            ]
        )


def test_source_and_keyword_inputs_are_mutually_exclusive():
    with pytest.raises(InputValidationError, match="source.*不能同时"):
        normalize_samples(
            source=[],
            rgb="rgb.png",
            depth="depth.png",
            intrinsics="K.npz",
            target="apple",
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_manifest_dataset.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rgbd_grasp_sdk.datasets'`.

- [ ] **Step 3: Implement dataset helpers**

Create `src/rgbd_grasp_sdk/datasets/manifest.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rgbd_grasp_sdk.errors import InputValidationError
from rgbd_grasp_sdk.types import CameraIntrinsics


SampleValue = str | Path | Any


@dataclass(frozen=True)
class GraspSample:
    id: str | None
    rgb: SampleValue
    depth: SampleValue
    intrinsics: CameraIntrinsics | str | Path
    target: str


def load_samples(path: str | Path) -> list[GraspSample]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise InputValidationError(f"manifest 文件不存在: {manifest_path}")

    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)

    return normalize_samples(raw, base_dir=manifest_path.parent)


def normalize_samples(
    source: Any = None,
    *,
    rgb: Any = None,
    depth: Any = None,
    intrinsics: Any = None,
    target: str | None = None,
    base_dir: str | Path | None = None,
) -> list[GraspSample]:
    keyword_values = (rgb, depth, intrinsics, target)
    has_keyword_input = any(value is not None for value in keyword_values)
    if source is not None and has_keyword_input:
        raise InputValidationError("source 与 rgb/depth/intrinsics/target 不能同时使用")

    if source is None:
        if not has_keyword_input:
            raise InputValidationError("必须提供 source 或单帧 rgb/depth/intrinsics/target")
        raw_samples = [
            {
                "rgb": rgb,
                "depth": depth,
                "intrinsics": intrinsics,
                "target": target,
            }
        ]
    elif isinstance(source, dict):
        raw_samples = [source]
    elif isinstance(source, list):
        raw_samples = source
    else:
        raise InputValidationError("source 必须是样本 dict 或样本 list")

    if not isinstance(raw_samples, list):
        raise InputValidationError("manifest 根节点必须是样本列表")

    root = Path(base_dir) if base_dir is not None else None
    return [_sample_from_mapping(index, item, root) for index, item in enumerate(raw_samples)]


def _sample_from_mapping(
    index: int,
    item: Any,
    base_dir: Path | None,
) -> GraspSample:
    if not isinstance(item, dict):
        raise InputValidationError(f"sample[{index}] 必须是对象")

    rgb = _required(item, index, "rgb")
    depth = _required(item, index, "depth")
    intrinsics = _required(item, index, "intrinsics")
    target = _required(item, index, "target")
    if not isinstance(target, str) or not target:
        raise InputValidationError(f"sample[{index}].target 必须是非空字符串")

    sample_id = item.get("id")
    if sample_id is not None and not isinstance(sample_id, str):
        raise InputValidationError(f"sample[{index}].id 必须是字符串")

    return GraspSample(
        id=sample_id,
        rgb=_resolve_path_like(rgb, base_dir),
        depth=_resolve_path_like(depth, base_dir),
        intrinsics=_resolve_path_like(intrinsics, base_dir),
        target=target,
    )


def _required(item: dict[str, Any], index: int, key: str) -> Any:
    value = item.get(key)
    if value is None:
        raise InputValidationError(f"sample[{index}].{key} 缺失")
    return value


def _resolve_path_like(value: Any, base_dir: Path | None) -> Any:
    if base_dir is None:
        return value
    if isinstance(value, str):
        path = Path(value)
        return path if path.is_absolute() else base_dir / path
    if isinstance(value, Path):
        return value if value.is_absolute() else base_dir / value
    return value
```

Create `src/rgbd_grasp_sdk/datasets/__init__.py`:

```python
from rgbd_grasp_sdk.datasets.manifest import GraspSample, load_samples, normalize_samples

__all__ = ["GraspSample", "load_samples", "normalize_samples"]
```

- [ ] **Step 4: Run manifest tests**

Run:

```bash
pytest tests/test_manifest_dataset.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rgbd_grasp_sdk/datasets tests/test_manifest_dataset.py
git commit -m "添加抓取样本manifest解析"
```

---

### Task 2: Validation Metrics Aggregation

**Files:**
- Create: `src/rgbd_grasp_sdk/evaluation/__init__.py`
- Create: `src/rgbd_grasp_sdk/evaluation/metrics.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing evaluation tests**

Create `tests/test_evaluation.py`:

```python
from __future__ import annotations

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
    assert summary["timings"]["total_mean"] == 0.3666666666666667
    assert summary["timings"]["segmentation_mean"] == 0.15000000000000002
    assert summary["timings"]["grasping_mean"] == 0.3


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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_evaluation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rgbd_grasp_sdk.evaluation'`.

- [ ] **Step 3: Implement validation metrics**

Create `src/rgbd_grasp_sdk/evaluation/metrics.py`:

```python
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
```

Create `src/rgbd_grasp_sdk/evaluation/__init__.py`:

```python
from rgbd_grasp_sdk.evaluation.metrics import summarize_validation

__all__ = ["summarize_validation"]
```

- [ ] **Step 4: Run evaluation tests**

Run:

```bash
pytest tests/test_evaluation.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rgbd_grasp_sdk/evaluation tests/test_evaluation.py
git commit -m "添加无标注评估指标聚合"
```

---

### Task 3: Benchmark Metrics Aggregation

**Files:**
- Create: `src/rgbd_grasp_sdk/benchmarking/__init__.py`
- Create: `src/rgbd_grasp_sdk/benchmarking/stats.py`
- Test: `tests/test_benchmarking.py`

- [ ] **Step 1: Write failing benchmark tests**

Create `tests/test_benchmarking.py`:

```python
from __future__ import annotations

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
    assert summary["elapsed"]["mean"] == 0.21
    assert summary["elapsed"]["p50"] == 0.21
    assert summary["elapsed"]["p95"] == 0.31
    assert summary["elapsed"]["max"] == 0.31
    assert summary["timings"]["total"]["mean"] == 0.19999999999999998
    assert summary["timings"]["grasping"]["mean"] == 0.095
    assert summary["samples_per_sec"] == 1 / 0.21
    assert summary["backend"] == {"segmentation": "yolo", "grasping": "rng"}


def test_summarize_benchmark_handles_no_records():
    summary = summarize_benchmark([], warmup=0, repeat=1, backend_summary={})

    assert summary["total"] == 0
    assert summary["elapsed"] == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    assert summary["samples_per_sec"] == 0.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_benchmarking.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rgbd_grasp_sdk.benchmarking'`.

- [ ] **Step 3: Implement benchmark stats**

Create `src/rgbd_grasp_sdk/benchmarking/stats.py`:

```python
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
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
        "mean": float(mean(ordered)),
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "max": ordered[-1],
    }


def _nearest_rank(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]
```

Create `src/rgbd_grasp_sdk/benchmarking/__init__.py`:

```python
from rgbd_grasp_sdk.benchmarking.stats import BenchmarkRecord, summarize_benchmark

__all__ = ["BenchmarkRecord", "summarize_benchmark"]
```

- [ ] **Step 4: Run benchmark tests**

Run:

```bash
pytest tests/test_benchmarking.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rgbd_grasp_sdk/benchmarking tests/test_benchmarking.py
git commit -m "添加benchmark统计聚合"
```

---

### Task 4: Top-Level `RGBDGrasp` Predict API

**Files:**
- Create: `src/rgbd_grasp_sdk/model.py`
- Modify: `src/rgbd_grasp_sdk/__init__.py`
- Test: `tests/test_model_api.py`

- [ ] **Step 1: Write failing model API tests**

Create `tests/test_model_api.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from rgbd_grasp_sdk import RGBDGrasp
from rgbd_grasp_sdk.config.schema import (
    GraspingConfig,
    MaskConfig,
    OutputConfig,
    RankingConfig,
    SdkConfig,
    SegmentationConfig,
)
from rgbd_grasp_sdk.errors import InputValidationError
from rgbd_grasp_sdk.types import (
    CameraIntrinsics,
    GraspCandidate,
    PipelineResult,
    PipelineStatus,
    Pose6D,
)


class FakePipeline:
    def __init__(self):
        self.calls = []

    def run(self, rgb, depth, intrinsics, target):
        self.calls.append((rgb, depth, intrinsics, target))
        if target == "raise":
            raise RuntimeError("boom")
        if target == "fail":
            return PipelineResult(status=PipelineStatus.FAILED)
        grasp = GraspCandidate(
            pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            score=0.9,
            center_px=(1, 1),
        )
        return PipelineResult(
            status=PipelineStatus.SUCCESS,
            best_grasp=grasp,
            candidate_grasps=[grasp],
            timings={"total": 0.1},
        )


def _config() -> SdkConfig:
    return SdkConfig(
        segmentation=SegmentationConfig(
            backend="mock_seg",
            options={"model_path": "seg.pt", "device": "cpu"},
        ),
        grasping=GraspingConfig(
            backend="mock_grasp",
            options={"checkpoint_path": "rng.ckpt", "device": "cpu"},
        ),
        mask=MaskConfig(),
        ranking=RankingConfig(backend="default"),
        outputs=OutputConfig(visualize_3d=False),
    )


def _model_with_fake_pipeline():
    fake = FakePipeline()
    model = RGBDGrasp(_config(), pipeline_builder=lambda config, visualize_3d=None: fake)
    return model, fake


def test_rgbd_grasp_is_exported_from_package():
    assert RGBDGrasp.__name__ == "RGBDGrasp"


def test_predict_one_accepts_array_inputs_and_returns_pipeline_result():
    model, fake = _model_with_fake_pipeline()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.uint16)
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)

    result = model.predict_one(rgb=rgb, depth=depth, intrinsics=intrinsics, target="apple")

    assert result.status is PipelineStatus.SUCCESS
    assert fake.calls == [(rgb, depth, intrinsics, "apple")]


def test_predict_returns_list_for_single_keyword_input():
    model, _ = _model_with_fake_pipeline()

    results = model.predict(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0),
        target="apple",
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].status is PipelineStatus.SUCCESS


def test_predict_processes_source_list_independently():
    model, fake = _model_with_fake_pipeline()
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    source = [
        {
            "id": "a",
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16),
            "intrinsics": intrinsics,
            "target": "apple",
        },
        {
            "id": "b",
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16),
            "intrinsics": intrinsics,
            "target": "fail",
        },
    ]

    results = model.predict(source=source)

    assert [item.status for item in results] == [
        PipelineStatus.SUCCESS,
        PipelineStatus.FAILED,
    ]
    assert len(fake.calls) == 2


def test_strict_true_reraises_exceptions():
    model, _ = _model_with_fake_pipeline()

    with pytest.raises(RuntimeError, match="boom"):
        model.predict_one(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=np.ones((2, 2), dtype=np.uint16),
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0),
            target="raise",
            strict=True,
        )


def test_strict_false_converts_exceptions_to_failed_result():
    model, _ = _model_with_fake_pipeline()

    result = model.predict_one(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0),
        target="raise",
        strict=False,
    )

    assert result.status is PipelineStatus.FAILED
    assert result.error is not None
    assert result.error.code == "runtime_error"
    assert "boom" in result.error.message


def test_predict_requires_target():
    model, _ = _model_with_fake_pipeline()

    with pytest.raises(InputValidationError, match="target"):
        model.predict_one(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=np.ones((2, 2), dtype=np.uint16),
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0),
            target="",
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_model_api.py -q
```

Expected: FAIL because `RGBDGrasp` is not exported or `model.py` does not exist.

- [ ] **Step 3: Implement `RGBDGrasp` predict API**

Create `src/rgbd_grasp_sdk/model.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_grasp_sdk.config.loader import load_config
from rgbd_grasp_sdk.config.schema import SdkConfig
from rgbd_grasp_sdk.datasets import GraspSample, load_samples, normalize_samples
from rgbd_grasp_sdk.errors import InputValidationError
from rgbd_grasp_sdk.grasping.factory import create_grasp_predictor
from rgbd_grasp_sdk.io import read_depth, read_intrinsics_npz, read_rgb
from rgbd_grasp_sdk.pipeline.grasp_pipeline import GraspPipeline
from rgbd_grasp_sdk.publishers import JsonFilePublisher, TransformJsonFilePublisher
from rgbd_grasp_sdk.ranking.factory import create_ranker
from rgbd_grasp_sdk.segmentation.factory import create_segmenter
from rgbd_grasp_sdk.types import CameraIntrinsics, PipelineError, PipelineResult, PipelineStatus


PipelineBuilder = Callable[[SdkConfig, bool | None], Any]


class RGBDGrasp:
    def __init__(
        self,
        config: str | Path | SdkConfig,
        *,
        pipeline_builder: PipelineBuilder | None = None,
    ) -> None:
        self.config_path = Path(config) if isinstance(config, (str, Path)) else None
        self.config = load_config(config) if isinstance(config, (str, Path)) else config
        self._pipeline_builder = pipeline_builder or self._default_pipeline_builder
        self._pipeline = self._pipeline_builder(self.config, None)

    def predict_one(
        self,
        *,
        rgb: Any,
        depth: Any,
        intrinsics: Any,
        target: str,
        strict: bool = True,
        visualize_3d: bool | None = None,
        output_json: str | Path | None = None,
        output_transform_json: str | Path | None = None,
    ) -> PipelineResult:
        sample = normalize_samples(
            source=None,
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            target=target,
        )[0]
        result = self._run_sample(sample, strict=strict, visualize_3d=visualize_3d)
        self._publish_outputs(result, output_json, output_transform_json)
        return result

    def predict(
        self,
        source: Any = None,
        *,
        rgb: Any = None,
        depth: Any = None,
        intrinsics: Any = None,
        target: str | None = None,
        strict: bool = False,
        visualize_3d: bool | None = None,
    ) -> list[PipelineResult]:
        samples = normalize_samples(
            source,
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            target=target,
        )
        return [
            self._run_sample(sample, strict=strict, visualize_3d=visualize_3d)
            for sample in samples
        ]

    def _run_sample(
        self,
        sample: GraspSample,
        *,
        strict: bool,
        visualize_3d: bool | None,
    ) -> PipelineResult:
        try:
            if not sample.target:
                raise InputValidationError("target 必须是非空字符串")
            pipeline = self._pipeline
            if visualize_3d is not None:
                pipeline = self._pipeline_builder(self.config, visualize_3d)
            return pipeline.run(
                rgb=self._load_rgb(sample.rgb),
                depth=self._load_depth(sample.depth),
                intrinsics=self._load_intrinsics(sample.intrinsics),
                target=sample.target,
            )
        except Exception as exc:
            if strict:
                raise
            return PipelineResult(
                status=PipelineStatus.FAILED,
                error=PipelineError(
                    code=_exception_code(exc),
                    message=str(exc),
                    detail={"exception_type": type(exc).__name__},
                ),
                metadata={"target": sample.target, "sample_id": sample.id},
            )

    def _load_rgb(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (str, Path)):
            return read_rgb(value)
        raise InputValidationError("rgb 必须是 np.ndarray 或图片路径")

    def _load_depth(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (str, Path)):
            return read_depth(value)
        raise InputValidationError("depth 必须是 np.ndarray 或图片路径")

    def _load_intrinsics(self, value: Any) -> CameraIntrinsics:
        if isinstance(value, CameraIntrinsics):
            return value
        if isinstance(value, (str, Path)):
            return read_intrinsics_npz(value)
        raise InputValidationError("intrinsics 必须是 CameraIntrinsics 或 npz 路径")

    def _publish_outputs(
        self,
        result: PipelineResult,
        output_json: str | Path | None,
        output_transform_json: str | Path | None,
    ) -> None:
        if output_json is not None:
            JsonFilePublisher(output_json).publish(result)
        if output_transform_json is not None:
            TransformJsonFilePublisher(output_transform_json).publish(result)

    def _default_pipeline_builder(
        self,
        config: SdkConfig,
        visualize_3d: bool | None,
    ) -> GraspPipeline:
        segmenter = create_segmenter(config.segmentation.backend, config.segmentation.options)
        grasp_predictor = create_grasp_predictor(config.grasping.backend, config.grasping.options)
        return GraspPipeline(
            segmenter=segmenter,
            grasp_predictor=grasp_predictor,
            ranker=create_ranker(config.ranking),
            mask_config=config.mask,
            visualize_3d=config.outputs.visualize_3d if visualize_3d is None else visualize_3d,
        )


def _exception_code(exc: Exception) -> str:
    name = type(exc).__name__
    snake = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0:
            snake.append("_")
        snake.append(char.lower())
    return "".join(snake)
```

Modify `src/rgbd_grasp_sdk/__init__.py`:

```python
from rgbd_grasp_sdk.config.loader import load_config
from rgbd_grasp_sdk.model import RGBDGrasp
from rgbd_grasp_sdk.pipeline.grasp_pipeline import GraspPipeline
from rgbd_grasp_sdk.types import (
    CameraIntrinsics,
    GraspCandidate,
    PipelineResult,
    PipelineStatus,
    Pose6D,
)

__all__ = [
    "CameraIntrinsics",
    "GraspCandidate",
    "GraspPipeline",
    "PipelineResult",
    "PipelineStatus",
    "Pose6D",
    "RGBDGrasp",
    "load_config",
]
```

- [ ] **Step 4: Run model API tests**

Run:

```bash
pytest tests/test_model_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Run related tests**

Run:

```bash
pytest tests/test_manifest_dataset.py tests/test_model_api.py tests/test_pipeline_contracts.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rgbd_grasp_sdk/model.py src/rgbd_grasp_sdk/__init__.py tests/test_model_api.py
git commit -m "添加RGBDGrasp推理API"
```

---

### Task 5: `info()`, `val()`, And `benchmark()` On `RGBDGrasp`

**Files:**
- Modify: `src/rgbd_grasp_sdk/model.py`
- Test: `tests/test_model_api.py`

- [ ] **Step 1: Add failing tests for info, val, and benchmark**

Append to `tests/test_model_api.py`:

```python
def test_info_reports_config_backends_and_dependency_status():
    model, _ = _model_with_fake_pipeline()

    info = model.info()

    assert info["segmentation"]["backend"] == "mock_seg"
    assert info["grasping"]["backend"] == "mock_grasp"
    assert info["ranking"]["backend"] == "default"
    assert info["devices"] == {"segmentation": "cpu", "grasping": "cpu"}
    assert "dependencies" in info
    assert info["paths"]["segmentation.model_path"]["exists"] is False
    assert info["paths"]["grasping.checkpoint_path"]["exists"] is False


def test_val_uses_manifest_samples_and_validation_summary():
    model, _ = _model_with_fake_pipeline()
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)

    summary = model.val(
        data=[
            {
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth": np.ones((2, 2), dtype=np.uint16),
                "intrinsics": intrinsics,
                "target": "apple",
            },
            {
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth": np.ones((2, 2), dtype=np.uint16),
                "intrinsics": intrinsics,
                "target": "fail",
            },
        ]
    )

    assert summary["total"] == 2
    assert summary["success"] == 1
    assert summary["failed"] == 1


def test_benchmark_runs_warmup_and_repeat_without_counting_warmup():
    model, fake = _model_with_fake_pipeline()
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    data = [
        {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16),
            "intrinsics": intrinsics,
            "target": "apple",
        }
    ]

    summary = model.benchmark(data=data, warmup=2, repeat=3)

    assert summary["warmup"] == 2
    assert summary["repeat"] == 3
    assert summary["total"] == 3
    assert summary["success"] == 3
    assert len(fake.calls) == 5
    assert summary["backend"]["segmentation"] == "mock_seg"
    assert summary["backend"]["grasping"] == "mock_grasp"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_model_api.py -q
```

Expected: FAIL because `RGBDGrasp.info`, `RGBDGrasp.val`, or `RGBDGrasp.benchmark` is missing.

- [ ] **Step 3: Implement methods in `model.py`**

Modify `src/rgbd_grasp_sdk/model.py` imports:

```python
import importlib.metadata
import importlib.util
import time
```

Add imports:

```python
from rgbd_grasp_sdk.benchmarking import BenchmarkRecord, summarize_benchmark
from rgbd_grasp_sdk.evaluation import summarize_validation
```

Add these methods inside `RGBDGrasp`:

```python
    def info(self) -> dict[str, Any]:
        return {
            "version": _package_version(),
            "segmentation": {
                "backend": self.config.segmentation.backend,
                "options": dict(self.config.segmentation.options),
            },
            "grasping": {
                "backend": self.config.grasping.backend,
                "options": dict(self.config.grasping.options),
            },
            "ranking": {
                "backend": self.config.ranking.backend,
                "top_k": self.config.ranking.top_k,
            },
            "devices": {
                "segmentation": self.config.segmentation.options.get("device"),
                "grasping": self.config.grasping.options.get("device"),
            },
            "dependencies": {
                "cv2": importlib.util.find_spec("cv2") is not None,
                "numpy": importlib.util.find_spec("numpy") is not None,
                "ultralytics": importlib.util.find_spec("ultralytics") is not None,
                "torch": importlib.util.find_spec("torch") is not None,
                "open3d": importlib.util.find_spec("open3d") is not None,
            },
            "paths": self._path_status(),
        }

    def val(self, data: Any) -> dict[str, Any]:
        samples = self._load_data_samples(data)
        results = [
            self._run_sample(sample, strict=False, visualize_3d=None)
            for sample in samples
        ]
        return summarize_validation(results)

    def benchmark(
        self,
        data: Any,
        *,
        warmup: int = 1,
        repeat: int = 3,
    ) -> dict[str, Any]:
        samples = self._load_data_samples(data)
        for _ in range(warmup):
            for sample in samples:
                self._run_sample(sample, strict=False, visualize_3d=None)

        records: list[BenchmarkRecord] = []
        for _ in range(repeat):
            for sample in samples:
                started = time.perf_counter()
                result = self._run_sample(sample, strict=False, visualize_3d=None)
                records.append(
                    BenchmarkRecord(
                        result=result,
                        elapsed=time.perf_counter() - started,
                    )
                )

        return summarize_benchmark(
            records,
            warmup=warmup,
            repeat=repeat,
            backend_summary=self._backend_summary(),
        )

    def _load_data_samples(self, data: Any) -> list[GraspSample]:
        if isinstance(data, (str, Path)):
            return load_samples(data)
        return normalize_samples(data)

    def _backend_summary(self) -> dict[str, Any]:
        return {
            "segmentation": self.config.segmentation.backend,
            "grasping": self.config.grasping.backend,
            "ranking": self.config.ranking.backend,
            "devices": {
                "segmentation": self.config.segmentation.options.get("device"),
                "grasping": self.config.grasping.options.get("device"),
            },
        }

    def _path_status(self) -> dict[str, dict[str, Any]]:
        paths: dict[str, dict[str, Any]] = {}
        for section_name, options in (
            ("segmentation", self.config.segmentation.options),
            ("grasping", self.config.grasping.options),
        ):
            for key in ("model_path", "checkpoint_path"):
                value = options.get(key)
                if value is None:
                    continue
                path = Path(value)
                paths[f"{section_name}.{key}"] = {
                    "path": str(path),
                    "exists": path.exists(),
                }
        return paths
```

Add helper function at module bottom:

```python
def _package_version() -> str:
    try:
        return importlib.metadata.version("rgbd-grasp-sdk")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"
```

- [ ] **Step 4: Run model API tests**

Run:

```bash
pytest tests/test_model_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Run aggregation and model tests together**

Run:

```bash
pytest tests/test_evaluation.py tests/test_benchmarking.py tests/test_model_api.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rgbd_grasp_sdk/model.py tests/test_model_api.py
git commit -m "添加RGBDGrasp评估和benchmark入口"
```

---

### Task 6: Reuse `RGBDGrasp` In The Existing CLI

**Files:**
- Modify: `examples/run_image_pair.py`
- Test: `tests/test_cli_contract.py`

- [ ] **Step 1: Inspect current CLI tests**

Run:

```bash
sed -n '1,260p' tests/test_cli_contract.py
```

Expected: Output shows existing CLI contract tests for argument parsing, result publishing, or CLI behavior. Keep those assertions intact.

- [ ] **Step 2: Add a CLI construction test**

Append this test to `tests/test_cli_contract.py`:

```python
def test_main_uses_rgbd_grasp_predict_one(monkeypatch, tmp_path, capsys):
    from examples import run_image_pair
    from rgbd_grasp_sdk.types import GraspCandidate, PipelineResult, PipelineStatus, Pose6D

    calls = {}

    class FakeModel:
        def __init__(self, config):
            calls["config"] = config

        def predict_one(
            self,
            *,
            rgb,
            depth,
            intrinsics,
            target,
            visualize_3d=None,
            output_json=None,
            output_transform_json=None,
        ):
            calls["predict_one"] = {
                "rgb": rgb,
                "depth": depth,
                "intrinsics": intrinsics,
                "target": target,
                "visualize_3d": visualize_3d,
                "output_json": output_json,
                "output_transform_json": output_transform_json,
            }
            grasp = GraspCandidate(
                pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
                score=0.9,
                center_px=(1, 2),
            )
            return PipelineResult(
                status=PipelineStatus.SUCCESS,
                best_grasp=grasp,
                timings={"total": 0.1},
            )

    monkeypatch.setattr(run_image_pair, "RGBDGrasp", FakeModel)
    run_image_pair.main(
        [
            "--config",
            "config.yaml",
            "--rgb",
            "rgb.png",
            "--depth",
            "depth.png",
            "--intrinsics",
            "K.npz",
            "--target",
            "apple",
            "--output-json",
            str(tmp_path / "result.json"),
            "--output-transform-json",
            str(tmp_path / "tf.json"),
            "--no-visualize-3d",
        ]
    )

    assert calls["config"] == "config.yaml"
    assert calls["predict_one"]["rgb"] == "rgb.png"
    assert calls["predict_one"]["depth"] == "depth.png"
    assert calls["predict_one"]["intrinsics"] == "K.npz"
    assert calls["predict_one"]["target"] == "apple"
    assert calls["predict_one"]["visualize_3d"] is False
    assert calls["predict_one"]["output_json"] == str(tmp_path / "result.json")
    assert calls["predict_one"]["output_transform_json"] == str(tmp_path / "tf.json")
    assert "status: success" in capsys.readouterr().out
```

- [ ] **Step 3: Run CLI test and verify it fails**

Run:

```bash
pytest tests/test_cli_contract.py -q
```

Expected: FAIL because `run_image_pair.main()` currently takes no `argv` parameter or does not use `RGBDGrasp`.

- [ ] **Step 4: Refactor CLI to use `RGBDGrasp`**

Modify `examples/run_image_pair.py` to this shape while preserving existing printed fields:

```python
from __future__ import annotations

import argparse
from typing import Sequence

from rgbd_grasp_sdk import RGBDGrasp
from rgbd_grasp_sdk.types import PipelineResult


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单帧 RGB-D 抓取预测")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--rgb", required=True, help="RGB 图像路径")
    parser.add_argument("--depth", required=True, help="depth 图像路径")
    parser.add_argument("--intrinsics", required=True, help="包含 K 的相机内参 npz")
    parser.add_argument("--target", required=True, help="目标类别或文本描述")
    parser.add_argument("--output-json", help="可选 JSON 结果输出路径")
    parser.add_argument("--output-transform-json", help="可选抓取 TF JSON 输出路径")
    parser.add_argument(
        "--visualize-3d",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否显示抓取3D可视化，默认使用配置文件 outputs.visualize_3d",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model = RGBDGrasp(args.config)
    result = model.predict_one(
        rgb=args.rgb,
        depth=args.depth,
        intrinsics=args.intrinsics,
        target=args.target,
        visualize_3d=args.visualize_3d,
        output_json=args.output_json,
        output_transform_json=args.output_transform_json,
    )
    _print_result(result)


def _print_result(result: PipelineResult) -> None:
    print(f"status: {result.status.value}")
    if result.best_grasp is not None:
        print(f"best_score: {result.best_grasp.score:.4f}")
        print(f"best_center_px: {result.best_grasp.center_px}")
        print(f"best_pose: {result.best_grasp.pose}")
    if result.error is not None:
        print(f"error: {result.error.code} - {result.error.message}")
    print(f"timings: {result.timings}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run CLI tests**

Run:

```bash
pytest tests/test_cli_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Run model and CLI tests**

Run:

```bash
pytest tests/test_model_api.py tests/test_cli_contract.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add examples/run_image_pair.py tests/test_cli_contract.py
git commit -m "让CLI复用RGBDGrasp API"
```

---

### Task 7: Full Verification And Documentation Touch-Up

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`

- [ ] **Step 1: Add README Python API example**

Add this section to `README.md` after the Single-Frame CLI section:

````markdown
## Python API

```python
from rgbd_grasp_sdk import RGBDGrasp

model = RGBDGrasp("configs/yolo_rng.yaml")
result = model.predict_one(
    rgb="data/rgb.png",
    depth="data/depth.png",
    intrinsics="data/camera_intrinsics.npz",
    target="apple",
)

print(result.status.value)
print(result.best_grasp)
```

For batch experiments, pass an explicit sample list or manifest to `predict()`, `val()`, or `benchmark()`.
````

- [ ] **Step 2: Add architecture note**

Add this paragraph to `docs/architecture.md` after the data flow section:

```markdown
## Python API Layer

`RGBDGrasp` 是推荐的 Python 顶层入口。它负责加载配置、创建 adapter、标准化输入、调用 `GraspPipeline`，并提供 `predict_one()`、`predict()`、`info()`、`val()` 和 `benchmark()`。底层 `pipeline` 仍只依赖抽象接口，真实模型接入仍必须放在 adapter 内。
```

- [ ] **Step 3: Run full test suite**

Run:

```bash
pytest -q
```

Expected: PASS with all tests passing.

- [ ] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 5: Commit docs and any verification fixes**

```bash
git add README.md docs/architecture.md
git commit -m "补充Python API文档"
```

---

## Self-Review

- Spec coverage:
  - Top-level `RGBDGrasp`: Task 4.
  - Array/path input and explicit sample lists: Tasks 1 and 4.
  - `predict_one()`, `predict()`, `info()`, `val()`, `benchmark()`: Tasks 4 and 5.
  - Unlabeled validation metrics: Task 2 and Task 5.
  - Benchmark metrics: Task 3 and Task 5.
  - Existing adapter/factory/pipeline/CLI preservation: Tasks 4 and 6.
  - Documentation: Task 7.
- Placeholder scan:
  - No placeholder markers, incomplete sections, or "similar to" references.
  - The CLI reuse task provides complete test code and expected failure/pass commands.
- Type consistency:
  - `RGBDGrasp.predict_one()` returns `PipelineResult`.
  - `RGBDGrasp.predict()` returns `list[PipelineResult]`.
  - `GraspSample` fields match manifest examples and model loading code.
  - `BenchmarkRecord` is defined before `RGBDGrasp.benchmark()` imports it.
