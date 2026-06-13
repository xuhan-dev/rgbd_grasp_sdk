# RGB-D Grasp SDK 第二阶段 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将第一阶段的占位 adapter 扩展为可接入真实 YOLO、FastSAM 和 RegionNormalizedGrasp 的 SDK，同时保持 pipeline 解耦、基础包轻量可导入。

**Architecture:** 第二阶段继续坚持 adapter 隔离：`pipeline` 只依赖 `Segmenter`、`GraspPredictor`、`GraspRanker` 抽象接口，真实模型依赖只允许出现在各自 adapter 内。默认测试仍不依赖 GPU 和模型权重；真实模型验证用手动 smoke test 或 `pytest -m integration`。

**Tech Stack:** Python 3.10、numpy、opencv-python、PyYAML、pytest、ultralytics、torch、RegionNormalizedGrasp 子模块、可选 FastSAM 源码路径。

---

## 文件结构变更

第二阶段主要修改和新增：

```text
pyproject.toml
requirements.txt
README.md
configs/
  yolo_rng.yaml
  fastsam_rng.yaml
  smoke_yolo_rng.yaml
docs/
  architecture.md
  model_adapters.md
examples/
  run_image_pair.py
src/rgbd_grasp_sdk/
  config/
    loader.py
    schema.py
  masks/
    postprocess.py
  segmentation/
    yolo_segmenter.py
    fastsam_segmenter.py
  grasping/
    rng_predictor.py
    rng_adapter_utils.py
  serialization/
    __init__.py
    result_json.py
  transforms/
    __init__.py
    pose.py
tests/
  test_optional_imports.py
  test_yolo_segmenter.py
  test_fastsam_segmenter.py
  test_rng_predictor.py
  test_mask_postprocess.py
  test_result_json.py
  test_cli_contract.py
```

职责边界：

- `segmentation/yolo_segmenter.py`：只负责 YOLO segmentation 推理和统一结果转换。
- `segmentation/fastsam_segmenter.py`：只负责 FastSAM 文本提示推理和统一结果转换。
- `grasping/rng_predictor.py`：只负责 RNG 模型加载、推理调用和 SDK 类型转换。
- `grasping/rng_adapter_utils.py`：放 RNG 输出转 `GraspCandidate` 的纯函数，便于测试。
- `masks/postprocess.py`：实现配置驱动的 mask 后处理。
- `transforms/pose.py`：姿态和 transform 辅助函数，不依赖 ROS/DDS。
- `serialization/result_json.py`：将 `PipelineResult` 转 JSON 友好的 dict。

---

### Task 1: 整理可选依赖和测试标记

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Create: `tests/test_optional_imports.py`

- [ ] **Step 1: 写入失败测试**

Create `tests/test_optional_imports.py`:

```python
def test_base_sdk_imports_without_constructing_model_backends():
    import rgbd_grasp_sdk
    from rgbd_grasp_sdk.pipeline.grasp_pipeline import GraspPipeline
    from rgbd_grasp_sdk.types import PipelineResult

    assert rgbd_grasp_sdk.PipelineResult is PipelineResult
    assert GraspPipeline.__name__ == "GraspPipeline"
```

- [ ] **Step 2: 运行测试确认当前基线**

Run:

```bash
pytest tests/test_optional_imports.py -q
```

Expected: PASS。若失败，先修复基础导入，不允许继续加重依赖。

- [ ] **Step 3: 修改 `pyproject.toml`**

Update `pyproject.toml` optional dependencies:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=7.4",
]
yolo = [
  "ultralytics>=8.3",
]
fastsam = [
  "torch>=1.11",
  "torchvision>=0.12",
  "ftfy",
  "regex",
  "tqdm",
]
rng = [
  "torch>=1.11",
  "torchvision>=0.12",
  "open3d",
  "transforms3d",
  "scikit-image",
  "trimesh",
]
all-models = [
  "rgbd-grasp-sdk[yolo,fastsam,rng]",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "integration: requires model weights or GPU-capable model dependencies",
]
```

Keep `[project.dependencies]` limited to lightweight dependencies:

```toml
dependencies = [
  "numpy>=1.23,<2.0",
  "opencv-python>=4.8",
  "PyYAML>=6.0",
]
```

- [ ] **Step 4: 修改 `requirements.txt`**

Keep `requirements.txt` lightweight:

```text
opencv-python>=4.8
PyYAML>=6.0
```

- [ ] **Step 5: 运行验证**

Run:

```bash
pytest -q
```

Expected: all tests PASS.

- [ ] **Step 6: 提交**

Run:

```bash
git add pyproject.toml requirements.txt tests/test_optional_imports.py
git commit -m "整理模型可选依赖和测试标记"
```

Expected: commit 成功。

---

### Task 2: 增强 mask 后处理

**Files:**
- Modify: `src/rgbd_grasp_sdk/masks/postprocess.py`
- Modify: `src/rgbd_grasp_sdk/config/schema.py`
- Test: `tests/test_mask_postprocess.py`

- [ ] **Step 1: 写入失败测试**

Create `tests/test_mask_postprocess.py`:

```python
import numpy as np

from rgbd_grasp_sdk.config.schema import MaskConfig
from rgbd_grasp_sdk.masks.postprocess import postprocess_masks
from rgbd_grasp_sdk.types import MaskResult


def test_postprocess_masks_merges_and_dilates():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True

    result = postprocess_masks(
        [MaskResult(mask=mask)],
        MaskConfig(merge_instances=True, dilate_kernel=3, dilate_iterations=1),
    )

    assert result.shape == (7, 7)
    assert result.sum() == 9


def test_postprocess_masks_removes_small_components():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = True
    mask[4:7, 4:7] = True

    result = postprocess_masks(
        [MaskResult(mask=mask)],
        MaskConfig(merge_instances=True, min_area=4),
    )

    assert not result[1, 1]
    assert result[5, 5]
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_mask_postprocess.py -q
```

Expected: FAIL with import error or missing `postprocess_masks`.

- [ ] **Step 3: 实现 `postprocess_masks`**

Modify `src/rgbd_grasp_sdk/masks/postprocess.py`:

```python
from __future__ import annotations

import cv2
import numpy as np

from rgbd_grasp_sdk.config.schema import MaskConfig
from rgbd_grasp_sdk.types import MaskResult


def merge_masks(masks: list[MaskResult]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)

    merged = np.zeros_like(masks[0].mask, dtype=bool)
    for item in masks:
        merged |= item.mask.astype(bool)
    return merged


def postprocess_masks(masks: list[MaskResult], config: MaskConfig) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)

    if config.merge_instances:
        result = merge_masks(masks)
    else:
        result = masks[0].mask.astype(bool)

    if config.min_area > 0:
        result = _remove_small_components(result, config.min_area)

    if config.dilate_kernel > 0 and config.dilate_iterations > 0:
        kernel = np.ones((config.dilate_kernel, config.dilate_kernel), dtype=np.uint8)
        result = cv2.dilate(result.astype(np.uint8), kernel, iterations=config.dilate_iterations).astype(bool)

    return result


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = True
    return cleaned
```

- [ ] **Step 4: 更新导出**

Modify `src/rgbd_grasp_sdk/masks/__init__.py`:

```python
from rgbd_grasp_sdk.masks.postprocess import merge_masks, postprocess_masks

__all__ = ["merge_masks", "postprocess_masks"]
```

- [ ] **Step 5: 运行测试**

Run:

```bash
pytest tests/test_mask_postprocess.py tests/test_mask_filter.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 6: 提交**

Run:

```bash
git add src/rgbd_grasp_sdk/masks tests/test_mask_postprocess.py
git commit -m "增强目标掩码后处理"
```

Expected: commit 成功。

---

### Task 3: 让 pipeline 使用配置化 mask 后处理

**Files:**
- Modify: `src/rgbd_grasp_sdk/pipeline/grasp_pipeline.py`
- Test: `tests/test_pipeline_contracts.py`

- [ ] **Step 1: 写入失败测试**

Append to `tests/test_pipeline_contracts.py`:

```python
from rgbd_grasp_sdk.config.schema import MaskConfig


def test_pipeline_applies_mask_postprocess_config():
    class OnePixelSegmenter:
        def segment(self, request: SegmentationRequest) -> SegmentationResult:
            mask = np.zeros(request.rgb.shape[:2], dtype=bool)
            mask[2, 2] = True
            return SegmentationResult(masks=[MaskResult(mask=mask)])

    class NearbyGraspPredictor:
        def predict(self, request: GraspRequest) -> GraspPredictionResult:
            return GraspPredictionResult(
                candidates=[
                    GraspCandidate(
                        pose=Pose6D(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
                        score=0.8,
                        center_px=(3, 2),
                    )
                ]
            )

    pipeline = GraspPipeline(
        segmenter=OnePixelSegmenter(),
        grasp_predictor=NearbyGraspPredictor(),
        mask_config=MaskConfig(dilate_kernel=3, dilate_iterations=1),
    )

    result = pipeline.run(
        rgb=np.zeros((6, 6, 3), dtype=np.uint8),
        depth=np.ones((6, 6), dtype=np.uint16),
        intrinsics=CameraIntrinsics(fx=600.0, fy=600.0, cx=3.0, cy=3.0),
        target="apple",
    )

    assert result.status is PipelineStatus.SUCCESS
    assert result.best_grasp is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_pipeline_contracts.py::test_pipeline_applies_mask_postprocess_config -q
```

Expected: FAIL because `GraspPipeline.__init__` does not accept `mask_config`.

- [ ] **Step 3: 修改 pipeline**

Modify `src/rgbd_grasp_sdk/pipeline/grasp_pipeline.py`:

```python
from rgbd_grasp_sdk.config.schema import MaskConfig
from rgbd_grasp_sdk.masks.postprocess import postprocess_masks
```

Update `__init__`:

```python
def __init__(
    self,
    segmenter: Segmenter,
    grasp_predictor: GraspPredictor,
    ranker: GraspRanker | None = None,
    mask_config: MaskConfig | None = None,
    min_grasp_score: float = 0.0,
) -> None:
    self.segmenter = segmenter
    self.grasp_predictor = grasp_predictor
    self.ranker = ranker or DefaultGraspRanker()
    self.mask_config = mask_config or MaskConfig()
    self.min_grasp_score = min_grasp_score
```

Replace:

```python
target_mask = merge_masks(segmentation.masks)
```

with:

```python
target_mask = postprocess_masks(segmentation.masks, self.mask_config)
```

- [ ] **Step 4: 运行测试**

Run:

```bash
pytest tests/test_pipeline_contracts.py tests/test_mask_postprocess.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 5: 提交**

Run:

```bash
git add src/rgbd_grasp_sdk/pipeline tests/test_pipeline_contracts.py
git commit -m "让流水线使用配置化掩码后处理"
```

Expected: commit 成功。

---

### Task 4: 实现 YOLO 分割 adapter

**Files:**
- Modify: `src/rgbd_grasp_sdk/segmentation/yolo_segmenter.py`
- Test: `tests/test_yolo_segmenter.py`

- [ ] **Step 1: 写入 mock 测试**

Create `tests/test_yolo_segmenter.py`:

```python
import numpy as np

from rgbd_grasp_sdk.segmentation.yolo_segmenter import YoloSegmenter
from rgbd_grasp_sdk.types import SegmentationRequest


class FakeMask:
    def __init__(self):
        self.data = np.array([[[0, 1], [0, 1]]], dtype=np.float32)


class FakeBoxes:
    cls = np.array([0])
    conf = np.array([0.75])


class FakeResult:
    names = {0: "apple"}
    masks = FakeMask()
    boxes = FakeBoxes()


class FakeModel:
    def __call__(self, image, **kwargs):
        return [FakeResult()]


def test_yolo_segmenter_converts_matching_class_to_mask():
    segmenter = YoloSegmenter({"model_path": "unused.pt"}, model=FakeModel())

    result = segmenter.segment(
        SegmentationRequest(rgb=np.zeros((2, 2, 3), dtype=np.uint8), target="apple")
    )

    assert len(result.masks) == 1
    assert result.masks[0].label == "apple"
    assert result.masks[0].score == 0.75
    assert result.masks[0].mask.dtype == bool
    assert result.masks[0].mask[:, 1].all()
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_yolo_segmenter.py -q
```

Expected: FAIL because `YoloSegmenter` still raises `BackendUnavailableError`.

- [ ] **Step 3: 实现 adapter**

Modify `src/rgbd_grasp_sdk/segmentation/yolo_segmenter.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np

from rgbd_grasp_sdk.errors import BackendUnavailableError, ConfigError
from rgbd_grasp_sdk.types import MaskResult, SegmentationRequest, SegmentationResult


class YoloSegmenter:
    def __init__(self, options: dict[str, Any], model: Any | None = None):
        self.options = options
        self.model = model or self._load_model(options)

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        results = self.model(
            request.rgb,
            conf=float(self.options.get("confidence", 0.25)),
            iou=float(self.options.get("iou", 0.6)),
            verbose=False,
        )
        masks: list[MaskResult] = []
        for result in results:
            masks.extend(self._extract_masks(result, request.target))
        return SegmentationResult(masks=masks, metadata={"backend": "yolo"})

    def _load_model(self, options: dict[str, Any]) -> Any:
        model_path = options.get("model_path")
        if not model_path:
            raise ConfigError("YoloSegmenter 需要 model_path")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendUnavailableError("缺少 ultralytics，请安装 rgbd-grasp-sdk[yolo]") from exc
        return YOLO(model_path)

    def _extract_masks(self, result: Any, target: str) -> list[MaskResult]:
        if getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
            return []

        mask_data = _to_numpy(result.masks.data)
        class_ids = _to_numpy(result.boxes.cls).astype(int)
        confidences = _to_numpy(result.boxes.conf)
        names = result.names

        masks: list[MaskResult] = []
        for index, class_id in enumerate(class_ids):
            label = names[int(class_id)]
            if label != target:
                continue
            masks.append(
                MaskResult(
                    mask=mask_data[index].astype(bool),
                    score=float(confidences[index]),
                    label=label,
                    metadata={"class_id": int(class_id)},
                )
            )
        return masks


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
```

- [ ] **Step 4: 运行测试**

Run:

```bash
pytest tests/test_yolo_segmenter.py tests/test_optional_imports.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 5: 提交**

Run:

```bash
git add src/rgbd_grasp_sdk/segmentation/yolo_segmenter.py tests/test_yolo_segmenter.py pyproject.toml
git commit -m "实现YOLO类别分割适配器"
```

Expected: commit 成功。

---

### Task 5: 增加 TF/JSON 输出辅助

**Files:**
- Create: `src/rgbd_grasp_sdk/transforms/__init__.py`
- Create: `src/rgbd_grasp_sdk/transforms/pose.py`
- Create: `src/rgbd_grasp_sdk/serialization/__init__.py`
- Create: `src/rgbd_grasp_sdk/serialization/result_json.py`
- Test: `tests/test_result_json.py`

- [ ] **Step 1: 写入失败测试**

Create `tests/test_result_json.py`:

```python
from rgbd_grasp_sdk.serialization.result_json import pipeline_result_to_dict
from rgbd_grasp_sdk.transforms.pose import grasp_to_transform
from rgbd_grasp_sdk.types import GraspCandidate, PipelineResult, PipelineStatus, Pose6D


def test_grasp_to_transform_uses_frame_names():
    grasp = GraspCandidate(
        pose=Pose6D(0.1, 0.2, 0.3, 0.0, 1.57, 3.14),
        score=0.8,
        center_px=(10, 20),
    )

    transform = grasp_to_transform(grasp, parent_frame="camera", child_frame="grasp")

    assert transform.parent_frame == "camera"
    assert transform.child_frame == "grasp"
    assert transform.pose is grasp.pose


def test_pipeline_result_to_dict_is_json_friendly():
    grasp = GraspCandidate(
        pose=Pose6D(0.1, 0.2, 0.3, 0.0, 1.57, 3.14),
        score=0.8,
        center_px=(10, 20),
    )
    result = PipelineResult(
        status=PipelineStatus.SUCCESS,
        best_grasp=grasp,
        candidate_grasps=[grasp],
        timings={"total": 0.01},
    )

    payload = pipeline_result_to_dict(result)

    assert payload["status"] == "success"
    assert payload["best_grasp"]["score"] == 0.8
    assert payload["best_grasp"]["pose"]["z"] == 0.3
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_result_json.py -q
```

Expected: FAIL with import error.

- [ ] **Step 3: 实现 transform helper**

Create `src/rgbd_grasp_sdk/transforms/pose.py`:

```python
from __future__ import annotations

from rgbd_grasp_sdk.types import GraspCandidate, Transform


def grasp_to_transform(
    grasp: GraspCandidate,
    parent_frame: str,
    child_frame: str,
) -> Transform:
    return Transform(parent_frame=parent_frame, child_frame=child_frame, pose=grasp.pose)
```

Create `src/rgbd_grasp_sdk/transforms/__init__.py`:

```python
from rgbd_grasp_sdk.transforms.pose import grasp_to_transform

__all__ = ["grasp_to_transform"]
```

- [ ] **Step 4: 实现 JSON 序列化**

Create `src/rgbd_grasp_sdk/serialization/result_json.py`:

```python
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
```

Create `src/rgbd_grasp_sdk/serialization/__init__.py`:

```python
from rgbd_grasp_sdk.serialization.result_json import pipeline_result_to_dict

__all__ = ["pipeline_result_to_dict"]
```

- [ ] **Step 5: 运行测试**

Run:

```bash
pytest tests/test_result_json.py -q
```

Expected: PASS.

- [ ] **Step 6: 提交**

Run:

```bash
git add src/rgbd_grasp_sdk/transforms src/rgbd_grasp_sdk/serialization tests/test_result_json.py
git commit -m "增加抓取结果JSON序列化"
```

Expected: commit 成功。

---

### Task 6: CLI 增加 JSON 输出

**Files:**
- Modify: `examples/run_image_pair.py`
- Test: `tests/test_cli_contract.py`

- [ ] **Step 1: 写入 CLI 参数测试**

Create `tests/test_cli_contract.py`:

```python
from examples.run_image_pair import parse_args


def test_parse_args_supports_output_json():
    args = parse_args(
        [
            "--config",
            "configs/yolo_rng.yaml",
            "--rgb",
            "data/rgb.png",
            "--depth",
            "data/depth.png",
            "--intrinsics",
            "data/camera_intrinsics.npz",
            "--target",
            "apple",
            "--output-json",
            "outputs/result.json",
        ]
    )

    assert args.output_json == "outputs/result.json"
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_cli_contract.py -q
```

Expected: FAIL because `parse_args` does not accept argv and has no `--output-json`.

- [ ] **Step 3: 修改 CLI**

Modify `examples/run_image_pair.py`:

```python
import json
from pathlib import Path
from typing import Sequence

from rgbd_grasp_sdk.serialization import pipeline_result_to_dict
```

Update parser:

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单帧 RGB-D 抓取预测")
    ...
    parser.add_argument("--output-json", help="可选 JSON 结果输出路径")
    return parser.parse_args(argv)
```

After pipeline result:

```python
if args.output_json:
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(pipeline_result_to_dict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
```

- [ ] **Step 4: 运行测试**

Run:

```bash
pytest tests/test_cli_contract.py -q
python examples/run_image_pair.py --help
```

Expected: pytest PASS and help output includes `--output-json`.

- [ ] **Step 5: 提交**

Run:

```bash
git add examples/run_image_pair.py tests/test_cli_contract.py
git commit -m "支持命令行JSON结果输出"
```

Expected: commit 成功。

---

### Task 7: 实现 RNG 输出转换纯函数

**Files:**
- Create: `src/rgbd_grasp_sdk/grasping/rng_adapter_utils.py`
- Test: `tests/test_rng_predictor.py`

- [ ] **Step 1: 写入转换测试**

Create `tests/test_rng_predictor.py`:

```python
import numpy as np

from rgbd_grasp_sdk.grasping.rng_adapter_utils import rng_grasp_to_candidate


class FakeRngGrasp:
    score = 0.7
    width = 0.05
    translation = np.array([0.1, 0.2, 0.3])
    rotation = np.eye(3)


def test_rng_grasp_to_candidate_converts_pose_and_center():
    candidate = rng_grasp_to_candidate(FakeRngGrasp(), center_px=(12, 34))

    assert candidate.score == 0.7
    assert candidate.width == 0.05
    assert candidate.center_px == (12, 34)
    assert candidate.pose.x == 0.1
    assert candidate.pose.z == 0.3
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_rng_predictor.py -q
```

Expected: FAIL with import error.

- [ ] **Step 3: 实现转换函数**

Create `src/rgbd_grasp_sdk/grasping/rng_adapter_utils.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np

from rgbd_grasp_sdk.types import GraspCandidate, Pose6D


def rng_grasp_to_candidate(rng_grasp: Any, center_px: tuple[int, int]) -> GraspCandidate:
    translation = np.asarray(rng_grasp.translation, dtype=float)
    roll, pitch, yaw = _rotation_matrix_to_euler_xyz(np.asarray(rng_grasp.rotation, dtype=float))
    return GraspCandidate(
        pose=Pose6D(
            x=float(translation[0]),
            y=float(translation[1]),
            z=float(translation[2]),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        ),
        score=float(getattr(rng_grasp, "score", 0.0)),
        center_px=center_px,
        width=float(getattr(rng_grasp, "width", 0.0)),
        metadata={"source": "rng"},
    )


def _rotation_matrix_to_euler_xyz(matrix: np.ndarray) -> tuple[float, float, float]:
    sy = float(np.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        pitch = float(np.arctan2(-matrix[2, 0], sy))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        roll = float(np.arctan2(-matrix[1, 2], matrix[1, 1]))
        pitch = float(np.arctan2(-matrix[2, 0], sy))
        yaw = 0.0
    return roll, pitch, yaw
```

- [ ] **Step 4: 运行测试**

Run:

```bash
pytest tests/test_rng_predictor.py -q
```

Expected: PASS.

- [ ] **Step 5: 提交**

Run:

```bash
git add src/rgbd_grasp_sdk/grasping/rng_adapter_utils.py tests/test_rng_predictor.py
git commit -m "增加RNG抓取结果转换工具"
```

Expected: commit 成功。

---

### Task 8: 实现 RNG adapter 骨架和延迟导入

**Files:**
- Modify: `src/rgbd_grasp_sdk/grasping/rng_predictor.py`
- Test: `tests/test_rng_predictor.py`

- [ ] **Step 1: 追加依赖缺失测试**

Append to `tests/test_rng_predictor.py`:

```python
import pytest

from rgbd_grasp_sdk.errors import BackendUnavailableError
from rgbd_grasp_sdk.grasping.rng_predictor import RngGraspPredictor


def test_rng_predictor_reports_missing_checkpoint_before_importing_heavy_model():
    with pytest.raises(BackendUnavailableError, match="checkpoint_path"):
        RngGraspPredictor({})
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_rng_predictor.py::test_rng_predictor_reports_missing_checkpoint_before_importing_heavy_model -q
```

Expected: FAIL because current placeholder does not validate `checkpoint_path`.

- [ ] **Step 3: 实现 RNG adapter 骨架**

Modify `src/rgbd_grasp_sdk/grasping/rng_predictor.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from rgbd_grasp_sdk.errors import BackendUnavailableError
from rgbd_grasp_sdk.types import GraspPredictionResult, GraspRequest


class RngGraspPredictor:
    def __init__(self, options: dict[str, Any], predictor: Any | None = None):
        self.options = options
        self.checkpoint_path = options.get("checkpoint_path")
        if not self.checkpoint_path:
            raise BackendUnavailableError("RngGraspPredictor 需要 checkpoint_path")
        self.predictor = predictor or self._load_predictor()

    def predict(self, request: GraspRequest) -> GraspPredictionResult:
        pred_gg, point_cloud = self.predictor.predict(request.rgb, request.depth, vis=False)
        candidates = []
        for grasp in pred_gg:
            center_px = tuple(int(v) for v in grasp.to_rect_grasp_group()[0].center)
            from rgbd_grasp_sdk.grasping.rng_adapter_utils import rng_grasp_to_candidate

            candidates.append(rng_grasp_to_candidate(grasp, center_px=center_px))
        return GraspPredictionResult(candidates=candidates, point_cloud=point_cloud, metadata={"backend": "rng"})

    def _load_predictor(self) -> Any:
        checkpoint_path = Path(str(self.checkpoint_path))
        if not checkpoint_path.exists():
            raise BackendUnavailableError(f"RNG checkpoint_path 不存在: {checkpoint_path}")
        try:
            from rgbd_grasp_sdk.grasping._rng_import import load_rng_predictor
        except ImportError as exc:
            raise BackendUnavailableError("无法导入 RNG 适配层依赖") from exc
        return load_rng_predictor(self.options)
```

Also create `src/rgbd_grasp_sdk/grasping/_rng_import.py`:

```python
from __future__ import annotations

from typing import Any


def load_rng_predictor(options: dict[str, Any]) -> Any:
    from preprocessor.RNG import GraspPredictor

    return GraspPredictor(
        checkpoint_path=str(options["checkpoint_path"]),
        device=str(options.get("device", "cuda")),
        input_size=tuple(options.get("input_size", [360, 640])),
        config=options.get("model_config", {}),
    )
```

- [ ] **Step 4: 运行测试**

Run:

```bash
pytest tests/test_rng_predictor.py tests/test_optional_imports.py -q
```

Expected: selected tests PASS.

- [ ] **Step 5: 提交**

Run:

```bash
git add src/rgbd_grasp_sdk/grasping/rng_predictor.py src/rgbd_grasp_sdk/grasping/_rng_import.py tests/test_rng_predictor.py
git commit -m "实现RNG适配器加载骨架"
```

Expected: commit 成功。

---

### Task 9: 实现 FastSAM adapter 骨架

**Files:**
- Modify: `src/rgbd_grasp_sdk/segmentation/fastsam_segmenter.py`
- Test: `tests/test_fastsam_segmenter.py`

- [ ] **Step 1: 写入 mock 测试**

Create `tests/test_fastsam_segmenter.py`:

```python
import numpy as np

from rgbd_grasp_sdk.segmentation.fastsam_segmenter import FastSamSegmenter
from rgbd_grasp_sdk.types import SegmentationRequest


class FakeFastSamBackend:
    def segment_text(self, image, text):
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[1, 1] = True
        return mask, image.copy()


def test_fastsam_segmenter_returns_text_prompt_mask():
    segmenter = FastSamSegmenter({"model_path": "unused.pt"}, backend=FakeFastSamBackend())

    result = segmenter.segment(
        SegmentationRequest(rgb=np.zeros((3, 3, 3), dtype=np.uint8), target="red apple")
    )

    assert len(result.masks) == 1
    assert result.masks[0].label == "red apple"
    assert result.masks[0].mask[1, 1]
    assert result.preview is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_fastsam_segmenter.py -q
```

Expected: FAIL because current placeholder raises `BackendUnavailableError`.

- [ ] **Step 3: 实现 FastSAM adapter 骨架**

Modify `src/rgbd_grasp_sdk/segmentation/fastsam_segmenter.py`:

```python
from __future__ import annotations

from typing import Any

from rgbd_grasp_sdk.errors import BackendUnavailableError, ConfigError
from rgbd_grasp_sdk.types import MaskResult, SegmentationRequest, SegmentationResult


class FastSamSegmenter:
    def __init__(self, options: dict[str, Any], backend: Any | None = None):
        self.options = options
        self.backend = backend or self._load_backend(options)

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        mask, preview = self.backend.segment_text(request.rgb, request.target)
        masks = []
        if mask is not None and mask.size > 0 and mask.any():
            masks.append(MaskResult(mask=mask.astype(bool), score=None, label=request.target))
        return SegmentationResult(masks=masks, preview=preview, metadata={"backend": "fastsam"})

    def _load_backend(self, options: dict[str, Any]) -> Any:
        model_path = options.get("model_path")
        if not model_path:
            raise ConfigError("FastSamSegmenter 需要 model_path")
        try:
            from rgbd_grasp_sdk.segmentation._fastsam_import import FastSamBackend
        except ImportError as exc:
            raise BackendUnavailableError("无法导入 FastSAM 适配层依赖") from exc
        return FastSamBackend(options)
```

Create `src/rgbd_grasp_sdk/segmentation/_fastsam_import.py`:

```python
from __future__ import annotations

from typing import Any


class FastSamBackend:
    def __init__(self, options: dict[str, Any]):
        from fastsam import FastSAM, FastSAMPrompt

        self.options = options
        self.model = FastSAM(options["model_path"])
        self.prompt_cls = FastSAMPrompt

    def segment_text(self, image, text):
        result = self.model(
            image,
            device=self.options.get("device", "cuda"),
            retina_masks=bool(self.options.get("retina", True)),
            imgsz=int(self.options.get("image_size", 1280)),
            conf=float(self.options.get("confidence", 0.2)),
            iou=float(self.options.get("iou", 0.6)),
            save=False,
        )
        prompt = self.prompt_cls(image, result, device=self.options.get("device", "cuda"))
        ann = prompt.text_prompt(text=text)
        preview = prompt.plot(annotations=ann, output_path=None)
        if ann is None or len(ann) == 0:
            return None, preview
        return ann[0].astype(bool), preview
```

- [ ] **Step 4: 运行测试**

Run:

```bash
pytest tests/test_fastsam_segmenter.py tests/test_optional_imports.py -q
```

Expected: selected tests PASS.

- [ ] **Step 5: 提交**

Run:

```bash
git add src/rgbd_grasp_sdk/segmentation/fastsam_segmenter.py src/rgbd_grasp_sdk/segmentation/_fastsam_import.py tests/test_fastsam_segmenter.py
git commit -m "实现FastSAM文本分割适配器骨架"
```

Expected: commit 成功。

---

### Task 10: 文档和 smoke test 说明

**Files:**
- Modify: `README.md`
- Create: `docs/model_adapters.md`
- Create: `configs/smoke_yolo_rng.yaml`

- [ ] **Step 1: 写模型适配文档**

Create `docs/model_adapters.md`:

```markdown
# 模型适配说明

## 基础原则

- 基础 SDK 不强制安装 torch、ultralytics、FastSAM 或 RNG 依赖。
- 真实模型依赖只在 adapter 内延迟导入。
- `pipeline` 不允许 import 具体模型类。

## YOLO

安装：

```bash
pip install -e ".[yolo]"
```

配置：

```yaml
segmentation:
  backend: yolo
  model_path: <legacy-project-root>/examples/weights/yolo11x-seg.pt
```

## RNG

安装：

```bash
pip install -e ".[rng]"
```

`RegionNormalizedGrasp` 通过 `third_party/RegionNormalizedGrasp` 子模块接入。

## FastSAM

FastSAM 依赖需要按实际环境配置源码路径或安装包。基础 SDK 不在 import 阶段加载 FastSAM。
```

- [ ] **Step 2: 写 smoke 配置**

Create `configs/smoke_yolo_rng.yaml`:

```yaml
segmentation:
  backend: yolo
  model_path: <legacy-project-root>/examples/weights/yolo11x-seg.pt
  device: cuda
  confidence: 0.25
  iou: 0.6

grasping:
  backend: rng
  checkpoint_path: <project-root>/third_party/RegionNormalizedGrasp/checkpoints/realsense
  device: cuda
  input_size: [360, 640]

mask:
  merge_instances: true
  dilate_kernel: 10
  dilate_iterations: 2
  min_area: 0

ranking:
  backend: default
  top_k: 10

outputs:
  return_point_cloud: false
  return_segmentation_preview: true
  return_candidates: true
```

- [ ] **Step 3: 更新 README**

Append to `README.md`:

```markdown
## 第二阶段模型接入

基础包仍保持轻量导入。真实模型依赖通过 extras 安装：

```bash
pip install -e ".[yolo]"
pip install -e ".[rng]"
```

单帧 smoke test 示例：

```bash
python examples/run_image_pair.py \
  --config configs/smoke_yolo_rng.yaml \
  --rgb data/rgb.png \
  --depth data/depth.png \
  --intrinsics data/camera_intrinsics.npz \
  --target apple \
  --output-json outputs/result.json
```
```

- [ ] **Step 4: 运行验证**

Run:

```bash
pytest -q
rg -n -F "/home" README.md docs configs src tests examples
```

Expected: pytest PASS; path scan has no output.

- [ ] **Step 5: 提交**

Run:

```bash
git add README.md docs/model_adapters.md configs/smoke_yolo_rng.yaml
git commit -m "补充模型适配和smoke测试说明"
```

Expected: commit 成功。

---

### Task 11: 第二阶段总验证

**Files:**
- Modify only if verification exposes defects.

- [ ] **Step 1: 基础测试**

Run:

```bash
pytest -q
```

Expected: all tests PASS.

- [ ] **Step 2: 基础包导入验证**

Run:

```bash
python3 -c "import rgbd_grasp_sdk; print(rgbd_grasp_sdk.PipelineResult)"
```

Expected: prints `PipelineResult` without importing model dependencies.

- [ ] **Step 3: pipeline 解耦扫描**

Run:

```bash
rg -n "YoloSegmenter|FastSamSegmenter|RngGraspPredictor|ultralytics|FastSAM|RegionNormalizedGrasp|torch" src/rgbd_grasp_sdk/pipeline
```

Expected: no output.

- [ ] **Step 4: 隐私路径扫描**

Run:

```bash
rg -n -F "/home" README.md docs configs src tests examples
rg -n -F "/Users" README.md docs configs src tests examples
rg -n -F "C:\\" README.md docs configs src tests examples
```

Expected: no output.

- [ ] **Step 5: CLI help 验证**

Run:

```bash
python3 examples/run_image_pair.py --help
```

Expected: output includes `--output-json`.

- [ ] **Step 6: Git 状态**

Run:

```bash
git status --short
```

Expected: no output.

- [ ] **Step 7: 如有修复则提交**

Run only if Step 1-6 required fixes:

```bash
git add .
git commit -m "完成第二阶段验证修复"
```

Expected: commit 成功。

---

## 自查结果

- 覆盖范围：计划覆盖第二阶段确认的 YOLO adapter、FastSAM adapter、RNG adapter、mask 后处理、TF/JSON 输出、CLI JSON、文档和总验证。
- 不覆盖范围：不实现 ROS2/DDS/HTTP 发布，不控制机械臂，不做训练脚本迁移。
- 解耦约束：所有真实模型依赖只允许在 adapter 或 `_import.py` 文件内导入；总验证包含 pipeline 扫描。
- 隐私约束：文档和配置继续使用 `<project-root>`、`<legacy-project-root>`，总验证包含绝对路径扫描。
- 测试策略：默认 `pytest -q` 不需要 GPU、torch、ultralytics 或真实权重；真实模型 smoke test 作为手动环境验证。
