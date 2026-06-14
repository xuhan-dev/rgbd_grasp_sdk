# Mask-Aware Grasp Postprocess And TF Publisher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不裁剪 RNG 场景输入、不增加服务化入口的前提下，增强目标 mask 对抓取候选的后处理/排序约束，并固化 TF 输出与 publisher 抽象。

**Architecture:** RNG 继续接收完整 RGB-D 场景以保留周边环境上下文；目标 mask 只参与候选归属评分、过滤和排序。输出层新增 publisher 抽象，但只提供本地 JSON/标准输出/TF message 构造，不绑定 HTTP、ROS2、DDS 或长期运行服务。

**Tech Stack:** Python dataclasses、NumPy、PyYAML、pytest、现有 `rgbd_grasp_sdk` pipeline、RegionNormalizedGrasp、PyTorch3D、grasp_nms。

---

## File Structure

- Modify: `src/rgbd_grasp_sdk/config/schema.py`
  - 增加 mask-aware ranking 配置字段。
- Modify: `src/rgbd_grasp_sdk/config/loader.py`
  - 加载 ranking 权重、阈值和过滤开关。
- Create: `src/rgbd_grasp_sdk/ranking/mask_aware_ranker.py`
  - 根据目标 mask 计算候选归属分数，并组合 RNG 原始分数。
- Modify: `src/rgbd_grasp_sdk/ranking/__init__.py`
  - 导出新 ranker。
- Modify: `src/rgbd_grasp_sdk/pipeline/grasp_pipeline.py`
  - 让 ranker 可接收 `target_mask` 上下文；保持兼容默认 ranker。
- Modify: `src/rgbd_grasp_sdk/ranking/base.py`
  - 将 rank 接口扩展为可选上下文参数。
- Create: `src/rgbd_grasp_sdk/publishers/base.py`
  - 定义 publisher 抽象。
- Create: `src/rgbd_grasp_sdk/publishers/transform_message.py`
  - 将 best grasp 转成外部系统可消费的 TF message dict。
- Create: `src/rgbd_grasp_sdk/publishers/json_file.py`
  - 复用现有 JSON 序列化能力写文件。
- Create: `src/rgbd_grasp_sdk/publishers/stdout.py`
  - 输出简洁 TF message 到标准输出。
- Create: `src/rgbd_grasp_sdk/publishers/__init__.py`
  - 导出 publisher 类型。
- Modify: `src/rgbd_grasp_sdk/serialization/result_json.py`
  - 增加 TF message 字段或辅助转换入口。
- Create: `docs/transform_contract.md`
  - 固化坐标系、单位、欧拉角顺序、夹爪 TCP 和 JSON 字段。
- Create: `docs/smoke_tests.md`
  - 记录真实 GPU smoke 流程。
- Create: `scripts/smoke_real_gpu.sh`
  - 提供可重复执行的真实依赖 smoke。
- Modify: `configs/yolo_rng.yaml`
  - 增加 `ranking.backend: mask_aware` 示例参数。
- Modify: `configs/smoke_yolo_rng.yaml`
  - 增加 smoke 使用的 mask-aware 参数。
- Test: `tests/test_mask_aware_ranker.py`
  - 覆盖 mask 归属评分、过滤和排序。
- Test: `tests/test_pipeline_contracts.py`
  - 覆盖 pipeline 传递 target mask 给 ranker。
- Test: `tests/test_publishers.py`
  - 覆盖 TF message、JSON publisher、stdout publisher。
- Test: `tests/test_config.py`
  - 覆盖 ranking 新配置加载。

---

### Task 1: 扩展 Ranking 配置

**Files:**
- Modify: `src/rgbd_grasp_sdk/config/schema.py`
- Modify: `src/rgbd_grasp_sdk/config/loader.py`
- Modify: `configs/yolo_rng.yaml`
- Modify: `configs/smoke_yolo_rng.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写配置加载失败/成功测试**

Add to `tests/test_config.py`:

```python
def test_load_mask_aware_ranking_options(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
segmentation:
  backend: yolo
  model_path: weights.pt
grasping:
  backend: rng
  checkpoint_path: checkpoints/rng
mask:
  merge_instances: true
ranking:
  backend: mask_aware
  top_k: 5
  min_target_score: 0.25
  require_center_in_mask: true
  weights:
    rng_score: 0.7
    target_score: 0.3
outputs:
  visualize_3d: false
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.ranking.backend == "mask_aware"
    assert config.ranking.top_k == 5
    assert config.ranking.min_target_score == 0.25
    assert config.ranking.require_center_in_mask is True
    assert config.ranking.weights == {"rng_score": 0.7, "target_score": 0.3}
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest -q tests/test_config.py::test_load_mask_aware_ranking_options
```

Expected: FAIL，`RankingConfig` 缺少新字段。

- [ ] **Step 3: 扩展 schema**

Modify `src/rgbd_grasp_sdk/config/schema.py`:

```python
@dataclass(frozen=True)
class RankingConfig:
    backend: str = "default"
    top_k: int = 10
    min_target_score: float = 0.0
    require_center_in_mask: bool = False
    weights: dict[str, float] = field(
        default_factory=lambda: {"rng_score": 1.0, "target_score": 0.0}
    )
```

- [ ] **Step 4: 扩展 loader**

Modify `_load_ranking` in `src/rgbd_grasp_sdk/config/loader.py`:

```python
def _load_ranking(data: Any) -> RankingConfig:
    backend, options = _require_backend("ranking", data)
    raw_weights = options.get("weights", {})
    if raw_weights is None:
        raw_weights = {}
    if not isinstance(raw_weights, dict):
        raise ConfigError("ranking.weights 必须是对象")
    weights = {
        "rng_score": float(raw_weights.get("rng_score", 1.0)),
        "target_score": float(raw_weights.get("target_score", 0.0)),
    }
    return RankingConfig(
        backend=backend,
        top_k=int(options.get("top_k", 10)),
        min_target_score=float(options.get("min_target_score", 0.0)),
        require_center_in_mask=bool(options.get("require_center_in_mask", False)),
        weights=weights,
    )
```

- [ ] **Step 5: 更新配置示例**

In `configs/yolo_rng.yaml` and `configs/smoke_yolo_rng.yaml`:

```yaml
ranking:
  backend: mask_aware
  top_k: 10
  min_target_score: 0.0
  require_center_in_mask: false
  weights:
    rng_score: 0.7
    target_score: 0.3
```

- [ ] **Step 6: 运行配置测试**

Run:

```bash
pytest -q tests/test_config.py
```

Expected: PASS.

- [ ] **Step 7: 提交**

```bash
git add src/rgbd_grasp_sdk/config/schema.py src/rgbd_grasp_sdk/config/loader.py configs/yolo_rng.yaml configs/smoke_yolo_rng.yaml tests/test_config.py
git commit -m "增加mask感知排序配置"
```

---

### Task 2: 实现 Mask-Aware Ranker

**Files:**
- Modify: `src/rgbd_grasp_sdk/ranking/base.py`
- Create: `src/rgbd_grasp_sdk/ranking/mask_aware_ranker.py`
- Modify: `src/rgbd_grasp_sdk/ranking/__init__.py`
- Test: `tests/test_mask_aware_ranker.py`

- [ ] **Step 1: 写 ranker 行为测试**

Create `tests/test_mask_aware_ranker.py`:

```python
import numpy as np

from rgbd_grasp_sdk.config.schema import RankingConfig
from rgbd_grasp_sdk.ranking.mask_aware_ranker import MaskAwareGraspRanker
from rgbd_grasp_sdk.types import GraspCandidate, Pose6D


def _candidate(score, center_px):
    return GraspCandidate(
        pose=Pose6D(x=0.0, y=0.0, z=0.5, roll=0.0, pitch=0.0, yaw=0.0),
        score=score,
        center_px=center_px,
        width=0.06,
    )


def test_mask_aware_ranker_prefers_candidate_inside_target_mask():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    outside_high_score = _candidate(0.95, (1, 1))
    inside_lower_score = _candidate(0.70, (10, 10))
    ranker = MaskAwareGraspRanker(
        RankingConfig(
            backend="mask_aware",
            weights={"rng_score": 0.5, "target_score": 0.5},
        )
    )

    ranked = ranker.rank([outside_high_score, inside_lower_score], target_mask=mask)

    assert ranked[0] is inside_lower_score
    assert ranked[0].metadata["target_score"] == 1.0
    assert ranked[1].metadata["target_score"] == 0.0


def test_mask_aware_ranker_can_require_center_inside_mask():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    ranker = MaskAwareGraspRanker(
        RankingConfig(
            backend="mask_aware",
            require_center_in_mask=True,
        )
    )

    ranked = ranker.rank([_candidate(0.95, (1, 1)), _candidate(0.70, (10, 10))], target_mask=mask)

    assert len(ranked) == 1
    assert ranked[0].center_px == (10, 10)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest -q tests/test_mask_aware_ranker.py
```

Expected: FAIL，模块不存在。

- [ ] **Step 3: 扩展 ranker base 接口**

Modify `src/rgbd_grasp_sdk/ranking/base.py`:

```python
from __future__ import annotations

from typing import Protocol

import numpy as np

from rgbd_grasp_sdk.types import GraspCandidate


class GraspRanker(Protocol):
    def rank(
        self,
        candidates: list[GraspCandidate],
        target_mask: np.ndarray | None = None,
    ) -> list[GraspCandidate]:
        ...
```

Also update `DefaultGraspRanker.rank` signature to accept `target_mask: np.ndarray | None = None` and ignore it.

- [ ] **Step 4: 实现 mask-aware ranker**

Create `src/rgbd_grasp_sdk/ranking/mask_aware_ranker.py`:

```python
from __future__ import annotations

import numpy as np

from rgbd_grasp_sdk.config.schema import RankingConfig
from rgbd_grasp_sdk.types import GraspCandidate


class MaskAwareGraspRanker:
    def __init__(self, config: RankingConfig):
        self.config = config

    def rank(
        self,
        candidates: list[GraspCandidate],
        target_mask: np.ndarray | None = None,
    ) -> list[GraspCandidate]:
        scored = [
            self._with_scores(candidate, target_mask)
            for candidate in candidates
        ]
        filtered = [
            candidate for candidate in scored
            if candidate.metadata["target_score"] >= self.config.min_target_score
        ]
        if self.config.require_center_in_mask:
            filtered = [
                candidate for candidate in filtered
                if candidate.metadata["center_in_mask"]
            ]
        return sorted(
            filtered,
            key=lambda candidate: candidate.metadata["final_score"],
            reverse=True,
        )[: self.config.top_k]

    def _with_scores(
        self,
        candidate: GraspCandidate,
        target_mask: np.ndarray | None,
    ) -> GraspCandidate:
        target_score = _target_score(candidate, target_mask)
        rng_weight = float(self.config.weights.get("rng_score", 1.0))
        target_weight = float(self.config.weights.get("target_score", 0.0))
        final_score = candidate.score * rng_weight + target_score * target_weight
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "center_in_mask": bool(target_score > 0.0),
                "target_score": float(target_score),
                "final_score": float(final_score),
            }
        )
        return GraspCandidate(
            pose=candidate.pose,
            score=candidate.score,
            center_px=candidate.center_px,
            width=candidate.width,
            metadata=metadata,
        )


def _target_score(
    candidate: GraspCandidate,
    target_mask: np.ndarray | None,
) -> float:
    if target_mask is None or target_mask.size == 0:
        return 0.0
    x, y = candidate.center_px
    if y < 0 or x < 0 or y >= target_mask.shape[0] or x >= target_mask.shape[1]:
        return 0.0
    return 1.0 if bool(target_mask[y, x]) else 0.0
```

- [ ] **Step 5: 导出新 ranker**

Modify `src/rgbd_grasp_sdk/ranking/__init__.py`:

```python
from rgbd_grasp_sdk.ranking.default_ranker import DefaultGraspRanker
from rgbd_grasp_sdk.ranking.mask_aware_ranker import MaskAwareGraspRanker

__all__ = ["DefaultGraspRanker", "MaskAwareGraspRanker"]
```

- [ ] **Step 6: 运行 ranker 测试**

Run:

```bash
pytest -q tests/test_mask_aware_ranker.py
```

Expected: PASS.

- [ ] **Step 7: 提交**

```bash
git add src/rgbd_grasp_sdk/ranking tests/test_mask_aware_ranker.py
git commit -m "实现mask感知抓取排序"
```

---

### Task 3: 将 Mask-Aware Ranker 接入 Pipeline

**Files:**
- Modify: `src/rgbd_grasp_sdk/ranking/default_ranker.py`
- Modify: `src/rgbd_grasp_sdk/ranking/factory.py` if created during execution; otherwise create it.
- Modify: `src/rgbd_grasp_sdk/pipeline/grasp_pipeline.py`
- Modify: `examples/run_image_pair.py`
- Test: `tests/test_pipeline_contracts.py`

- [ ] **Step 1: 写 pipeline 传递 target mask 的测试**

Add to `tests/test_pipeline_contracts.py`:

```python
class RecordingRanker:
    def __init__(self):
        self.received_mask = None

    def rank(self, candidates, target_mask=None):
        self.received_mask = target_mask
        return candidates


def test_pipeline_passes_target_mask_to_ranker():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    ranker = RecordingRanker()
    pipeline = GraspPipeline(
        segmenter=FakeSegmenter(mask),
        grasp_predictor=FakeGraspPredictor(
            [
                GraspCandidate(
                    pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
                    score=0.8,
                    center_px=(1, 1),
                )
            ]
        ),
        ranker=ranker,
    )

    pipeline.run(
        rgb=rgb,
        depth=depth,
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
        target="target",
    )

    assert ranker.received_mask is not None
    assert ranker.received_mask[1, 1]
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest -q tests/test_pipeline_contracts.py::test_pipeline_passes_target_mask_to_ranker
```

Expected: FAIL，pipeline 尚未传递 `target_mask` 给 ranker。

- [ ] **Step 3: 修改 pipeline 调用**

Modify in `src/rgbd_grasp_sdk/pipeline/grasp_pipeline.py`:

```python
ranked = self.ranker.rank(filtered, target_mask=target_mask)
```

- [ ] **Step 4: 创建 ranker factory**

Create `src/rgbd_grasp_sdk/ranking/factory.py`:

```python
from __future__ import annotations

from rgbd_grasp_sdk.config.schema import RankingConfig
from rgbd_grasp_sdk.errors import ConfigError
from rgbd_grasp_sdk.ranking.base import GraspRanker
from rgbd_grasp_sdk.ranking.default_ranker import DefaultGraspRanker
from rgbd_grasp_sdk.ranking.mask_aware_ranker import MaskAwareGraspRanker


def create_ranker(config: RankingConfig) -> GraspRanker:
    if config.backend == "default":
        return DefaultGraspRanker(top_k=config.top_k)
    if config.backend == "mask_aware":
        return MaskAwareGraspRanker(config)
    raise ConfigError(f"不支持的 ranking backend: {config.backend}")
```

- [ ] **Step 5: CLI 使用 ranker factory**

Modify `examples/run_image_pair.py`:

```python
from rgbd_grasp_sdk.ranking.factory import create_ranker
```

and:

```python
pipeline = GraspPipeline(
    segmenter=segmenter,
    grasp_predictor=grasp_predictor,
    ranker=create_ranker(config.ranking),
    mask_config=config.mask,
    visualize_3d=visualize_3d,
)
```

- [ ] **Step 6: 运行 pipeline 测试**

Run:

```bash
pytest -q tests/test_pipeline_contracts.py tests/test_mask_aware_ranker.py
```

Expected: PASS.

- [ ] **Step 7: 提交**

```bash
git add src/rgbd_grasp_sdk/pipeline/grasp_pipeline.py src/rgbd_grasp_sdk/ranking examples/run_image_pair.py tests/test_pipeline_contracts.py
git commit -m "接入mask感知排序到pipeline"
```

---

### Task 4: 固化 TF 输出合约

**Files:**
- Create: `docs/transform_contract.md`
- Modify: `src/rgbd_grasp_sdk/types.py`
- Test: no runtime test required for docs, but run full docs path scan.

- [ ] **Step 1: 创建坐标系合约文档**

Create `docs/transform_contract.md`:

```markdown
# Transform Contract

本项目输出的抓取位姿用于外部系统消费。除非适配器另有说明，所有长度单位均为米，角度单位均为弧度。

## Frames

- `parent_frame`: 默认 `camera_color_optical_frame`。
- `child_frame`: 默认 `grasp_tcp`。

## Pose

`Pose6D` 字段：

- `x`, `y`, `z`: 抓取 TCP 在 `parent_frame` 下的位置，单位米。
- `roll`, `pitch`, `yaw`: 抓取 TCP 姿态，XYZ 欧拉角，单位弧度。

## Gripper Axes

- `approach_axis`: 夹爪接近目标的方向。
- `closing_axis`: 两指闭合方向。
- `tcp`: 夹爪工具中心点，用于外部系统执行抓取。

## Message

TF message JSON 格式：

```json
{
  "parent_frame": "camera_color_optical_frame",
  "child_frame": "grasp_tcp",
  "translation": {"x": 0.0, "y": 0.0, "z": 0.5},
  "rotation_rpy": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
  "score": 0.8,
  "center_px": [320, 240],
  "width": 0.06
}
```
```

- [ ] **Step 2: 给 Transform 类型增加默认 frame 约定**

Keep `Transform` in `src/rgbd_grasp_sdk/types.py` unchanged if it already matches:

```python
@dataclass(frozen=True)
class Transform:
    parent_frame: str
    child_frame: str
    pose: Pose6D
```

- [ ] **Step 3: 检查文档无隐私路径**

Run:

```bash
rg "/home/|Projects/" docs src tests
```

Expected: no matches.

- [ ] **Step 4: 提交**

```bash
git add docs/transform_contract.md src/rgbd_grasp_sdk/types.py
git commit -m "固化抓取TF输出合约"
```

---

### Task 5: 新增 Publisher 抽象，不增加服务化入口

**Files:**
- Create: `src/rgbd_grasp_sdk/publishers/base.py`
- Create: `src/rgbd_grasp_sdk/publishers/transform_message.py`
- Create: `src/rgbd_grasp_sdk/publishers/json_file.py`
- Create: `src/rgbd_grasp_sdk/publishers/stdout.py`
- Create: `src/rgbd_grasp_sdk/publishers/__init__.py`
- Test: `tests/test_publishers.py`

- [ ] **Step 1: 写 publisher 测试**

Create `tests/test_publishers.py`:

```python
import json

from rgbd_grasp_sdk.publishers.json_file import JsonFilePublisher
from rgbd_grasp_sdk.publishers.stdout import StdoutPublisher
from rgbd_grasp_sdk.publishers.transform_message import best_grasp_to_transform_message
from rgbd_grasp_sdk.types import GraspCandidate, PipelineResult, PipelineStatus, Pose6D


def _result():
    return PipelineResult(
        status=PipelineStatus.SUCCESS,
        best_grasp=GraspCandidate(
            pose=Pose6D(x=0.1, y=0.2, z=0.3, roll=0.0, pitch=0.1, yaw=0.2),
            score=0.9,
            center_px=(320, 240),
            width=0.06,
        ),
    )


def test_best_grasp_to_transform_message():
    message = best_grasp_to_transform_message(_result())

    assert message["parent_frame"] == "camera_color_optical_frame"
    assert message["child_frame"] == "grasp_tcp"
    assert message["translation"] == {"x": 0.1, "y": 0.2, "z": 0.3}
    assert message["rotation_rpy"] == {"roll": 0.0, "pitch": 0.1, "yaw": 0.2}
    assert message["score"] == 0.9
    assert message["center_px"] == [320, 240]
    assert message["width"] == 0.06


def test_json_file_publisher_writes_result(tmp_path):
    output_path = tmp_path / "result.json"
    JsonFilePublisher(output_path).publish(_result())

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["status"] == "success"
    assert data["best_grasp"]["score"] == 0.9


def test_stdout_publisher_prints_transform_message(capsys):
    StdoutPublisher().publish(_result())

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["child_frame"] == "grasp_tcp"
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest -q tests/test_publishers.py
```

Expected: FAIL，publisher 模块不存在。

- [ ] **Step 3: 实现 base**

Create `src/rgbd_grasp_sdk/publishers/base.py`:

```python
from __future__ import annotations

from typing import Protocol

from rgbd_grasp_sdk.types import PipelineResult


class GraspPublisher(Protocol):
    def publish(self, result: PipelineResult) -> None:
        ...
```

- [ ] **Step 4: 实现 transform message**

Create `src/rgbd_grasp_sdk/publishers/transform_message.py`:

```python
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
```

- [ ] **Step 5: 实现 JSON publisher**

Create `src/rgbd_grasp_sdk/publishers/json_file.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from rgbd_grasp_sdk.serialization import pipeline_result_to_dict
from rgbd_grasp_sdk.types import PipelineResult


class JsonFilePublisher:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def publish(self, result: PipelineResult) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(pipeline_result_to_dict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
```

- [ ] **Step 6: 实现 stdout publisher**

Create `src/rgbd_grasp_sdk/publishers/stdout.py`:

```python
from __future__ import annotations

import json

from rgbd_grasp_sdk.publishers.transform_message import best_grasp_to_transform_message
from rgbd_grasp_sdk.types import PipelineResult


class StdoutPublisher:
    def publish(self, result: PipelineResult) -> None:
        print(json.dumps(best_grasp_to_transform_message(result), ensure_ascii=False))
```

- [ ] **Step 7: 导出 publisher**

Create `src/rgbd_grasp_sdk/publishers/__init__.py`:

```python
from rgbd_grasp_sdk.publishers.base import GraspPublisher
from rgbd_grasp_sdk.publishers.json_file import JsonFilePublisher
from rgbd_grasp_sdk.publishers.stdout import StdoutPublisher
from rgbd_grasp_sdk.publishers.transform_message import best_grasp_to_transform_message

__all__ = [
    "GraspPublisher",
    "JsonFilePublisher",
    "StdoutPublisher",
    "best_grasp_to_transform_message",
]
```

- [ ] **Step 8: 运行 publisher 测试**

Run:

```bash
pytest -q tests/test_publishers.py
```

Expected: PASS.

- [ ] **Step 9: 提交**

```bash
git add src/rgbd_grasp_sdk/publishers tests/test_publishers.py
git commit -m "增加抓取结果发布抽象"
```

---

### Task 6: Smoke 脚本和文档

**Files:**
- Create: `scripts/smoke_real_gpu.sh`
- Create: `docs/smoke_tests.md`
- Modify: `README.md`

- [ ] **Step 1: 创建 smoke 脚本**

Create `scripts/smoke_real_gpu.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import torch
import pytorch3d
import grasp_nms

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("pytorch3d", getattr(pytorch3d, "__version__", "unknown"))
print("grasp_nms", getattr(grasp_nms, "__version__", "unknown"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY

PYTHONPATH="../sam_rng:../sam_rng/RegionNormalizedGrasp" \
python3 examples/run_image_pair.py \
  --config configs/smoke_yolo_rng.yaml \
  --rgb third_party/RegionNormalizedGrasp/images/demo_rgb.png \
  --depth third_party/RegionNormalizedGrasp/images/demo_depth.png \
  --intrinsics /tmp/rgbd_grasp_smoke/camera_intrinsics.npz \
  --target microwave \
  --output-json /tmp/rgbd_grasp_smoke/result.json \
  --no-visualize-3d
```

- [ ] **Step 2: 设置脚本权限**

Run:

```bash
chmod +x scripts/smoke_real_gpu.sh
```

- [ ] **Step 3: 创建 smoke 文档**

Create `docs/smoke_tests.md`:

```markdown
# Smoke Tests

## Unit Tests

```bash
pytest -q
```

## Real GPU Smoke

真实 smoke 用于确认以下依赖可用：

- CUDA
- PyTorch
- PyTorch3D
- grasp_nms
- RegionNormalizedGrasp 权重
- YOLO segmentation 权重

运行：

```bash
scripts/smoke_real_gpu.sh
```

预期输出包含：

```text
status: success
best_score:
best_center_px:
```

该 smoke 默认使用 `--no-visualize-3d`，避免 Open3D GUI 阻塞自动化检查。
```

- [ ] **Step 4: README 增加链接**

Add to `README.md`:

```markdown
## Smoke Tests

详见 `docs/smoke_tests.md`。
```

- [ ] **Step 5: 运行脚本语法检查和测试**

Run:

```bash
bash -n scripts/smoke_real_gpu.sh
pytest -q
```

Expected: both PASS.

- [ ] **Step 6: 运行真实 GPU smoke**

Run:

```bash
scripts/smoke_real_gpu.sh
```

Expected: output includes `status: success`.

- [ ] **Step 7: 提交**

```bash
git add scripts/smoke_real_gpu.sh docs/smoke_tests.md README.md
git commit -m "补充真实GPU smoke脚本"
```

---

### Task 7: 最终验证

**Files:**
- No new files.

- [ ] **Step 1: 全量测试**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: 空白检查**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 3: 隐私路径扫描**

Run:

```bash
rg "/home/|Projects/" README.md docs src tests configs scripts
```

Expected: no matches.

- [ ] **Step 4: 真实 GPU smoke**

Run:

```bash
scripts/smoke_real_gpu.sh
```

Expected: output includes `status: success`.

- [ ] **Step 5: 汇总状态**

Run:

```bash
git status --short
git log --oneline -8
```

Expected: clean worktree and latest commits are Chinese commit messages from this plan.

---

## Out Of Scope

- 不增加 HTTP、ZeroMQ、ROS2、DDS 或任何长期运行服务入口。
- 不裁剪 RNG 输入图像或点云。
- 不修改 RegionNormalizedGrasp 原始模型结构。
- 不引入新的深度学习模型训练流程。

## Self-Review

- Spec coverage: 覆盖 mask-aware 后处理/排序、TF 输出合约、publisher 抽象、真实 smoke 文档；明确排除服务化入口。
- Placeholder scan: 无 TBD、TODO、implement later。
- Type consistency: `RankingConfig`、`MaskAwareGraspRanker.rank(..., target_mask=None)`、`PipelineResult`、`GraspCandidate`、`Pose6D` 在任务间保持一致。
