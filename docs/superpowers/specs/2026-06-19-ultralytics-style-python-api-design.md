# Ultralytics-Style Python API Design

## Goal

为 RGB-D Grasp SDK 增加一个参考 Ultralytics 使用体验的 Python 顶层 API，让研究用户和集成用户可以通过一个稳定对象完成推理、无标注工程评估和性能 benchmark。

目标 API 优先服务 Python import 场景，不优先扩展 CLI、训练、模型导出、ROS2、HTTP、DDS 或长期运行服务。

## Scope

本轮设计包含：

1. 新增顶层 `RGBDGrasp` 对象，作为推荐 Python API。
2. 支持数组和文件路径混合输入。
3. 提供 `predict_one()`、`predict()`、`info()`、`val()`、`benchmark()`。
4. 为 `val()` 定义无标注工程评估指标。
5. 为 `benchmark()` 定义性能统计指标。
6. 保留现有 adapter、factory、`GraspPipeline` 和 CLI 行为。

不包含：

- 训练或微调入口。
- 神经网络权重导出。
- ROS2、HTTP、DDS、ZeroMQ 等服务化入口。
- 机械臂控制。
- 带抓取真值标注的模型准确性评测。
- 自动推断复杂数据集目录结构。

## API Shape

新增 `src/rgbd_grasp_sdk/model.py`，定义 `RGBDGrasp`，并从 `src/rgbd_grasp_sdk/__init__.py` 导出：

```python
from rgbd_grasp_sdk import RGBDGrasp

model = RGBDGrasp("configs/yolo_rng.yaml")
result = model.predict_one(
    rgb="data/rgb.png",
    depth="data/depth.png",
    intrinsics="data/camera_intrinsics.npz",
    target="apple",
)
```

`RGBDGrasp` 是薄封装：

- 初始化时加载 YAML 配置。
- 复用现有 `create_segmenter()`、`create_grasp_predictor()`、`create_ranker()`。
- 内部构建并调用 `GraspPipeline`。
- 不把分割、抓取、排序、发布逻辑揉进大类。

高级用户仍可直接使用现有 `GraspPipeline`、adapter 和 factory。

## Methods

### `predict_one()`

面向单帧输入，返回一个 `PipelineResult`。

支持：

- `rgb`: `np.ndarray` 或图片路径。
- `depth`: `np.ndarray` 或 depth 图片路径。
- `intrinsics`: `CameraIntrinsics`、`.npz` 路径，后续可扩展 3x3 `np.ndarray`。
- `target`: 必填字符串。
- `strict`: 默认 `True`。

第一版可支持少量方法级覆盖参数，例如 `visualize_3d`、`output_json`、`output_transform_json`。不要把全部 YAML 字段复制成函数参数。

### `predict()`

面向批量输入，返回 `list[PipelineResult]`。

第一版支持显式样本列表：

```python
results = model.predict(
    source=[
        {
            "id": "sample-a",
            "rgb": "a/rgb.png",
            "depth": "a/depth.png",
            "intrinsics": "a/camera_intrinsics.npz",
            "target": "apple",
        },
        {
            "id": "sample-b",
            "rgb": "b/rgb.png",
            "depth": "b/depth.png",
            "intrinsics": "b/camera_intrinsics.npz",
            "target": "banana",
        },
    ]
)
```

也可接受单帧关键字参数，并统一返回列表。

### `info()`

返回配置和运行环境摘要：

- SDK 版本。
- segmentation backend 和关键 options。
- grasping backend 和关键 options。
- ranking backend。
- device 配置。
- 可选依赖可用性。
- 关键模型或 checkpoint 路径是否存在。

### `val()`

面向无标注工程评估，输入 manifest 文件或样本列表。

第一版 manifest 是 YAML 或 JSON 列表，每条样本包含：

- `rgb`
- `depth`
- `intrinsics`
- `target`
- 可选 `id`

`val()` 不评价抓取真值准确性，只统计 pipeline 是否稳定可跑。

### `benchmark()`

复用 manifest 或样本列表，关注性能。

支持：

- `warmup`: 预热次数。
- `repeat`: 每个样本重复次数。
- `strict=False`: 默认不中断整批 benchmark。

## Result Strategy

第一版返回对象复用现有 `PipelineResult`，避免无意义迁移。后续如需更接近 Ultralytics，可增加 `GraspResults` 包装器，但不在第一版强制替换。

`predict_one()` 返回单个 `PipelineResult`。`predict()` 返回 `list[PipelineResult]`，即使输入只有一帧。

## Input Normalization

新增输入标准化层，职责是把用户输入转换成现有 pipeline 需要的对象：

- 图片路径通过 `read_rgb()` 和 `read_depth()` 读取。
- `.npz` 内参通过 `read_intrinsics_npz()` 读取。
- `CameraIntrinsics` 原样传递。
- 相对路径按 manifest 文件所在目录解析。
- 字段缺失时抛出清晰配置或数据错误。

该层不做模型推理，不做评估聚合。

## Error Handling

业务失败继续使用结构化 `PipelineResult`：

- `shape_mismatch`
- `empty_mask`
- `no_valid_grasp`

异常处理由 `strict` 控制：

- `strict=True`: Python 异常直接抛出，适合研究调试。
- `strict=False`: 异常转换为 `PipelineResult(status=FAILED, error=PipelineError(...))`，适合 `predict()`、`val()`、`benchmark()`。

批量执行时每条样本独立失败，不中断整批。

## Validation Metrics

`val()` 第一版输出无标注工程指标：

- `total`
- `success`
- `failed`
- `success_rate`
- `failure_reasons`
- `candidate_count.mean/min/max`
- `best_score.mean/min/max`
- `timings` 中各阶段 mean

失败样本不参与 `candidate_count` 和 `best_score` 的成功样本统计，但会参与失败率和失败原因统计。

## Benchmark Metrics

`benchmark()` 输出：

- warmup 和 repeat 配置。
- 总耗时 mean、p50、p95、max。
- segmentation、grasping、filtering、ranking 等阶段耗时统计。
- samples/sec 或 FPS。
- 成功率和失败原因。
- 后端和 device 摘要。

性能报告必须包含失败率，避免只展示成功样本的速度。

## Module Layout

建议新增：

```text
src/rgbd_grasp_sdk/model.py
src/rgbd_grasp_sdk/datasets/
src/rgbd_grasp_sdk/evaluation/
src/rgbd_grasp_sdk/benchmarking/
```

职责：

- `model.py`: `RGBDGrasp` 顶层对象。
- `datasets/`: manifest 解析、样本结构、路径解析。
- `evaluation/`: `val()` 指标聚合。
- `benchmarking/`: warmup、repeat、耗时分布和吞吐统计。

当前 `examples/run_image_pair.py` 第一阶段不强制改写。后续可让它复用 `RGBDGrasp.predict_one()`，但保持现有命令行行为。

## Testing

新增测试：

- `tests/test_model_api.py`: 初始化、`predict_one()`、`predict()`、数组/路径混合输入、`strict` 行为。
- `tests/test_manifest_dataset.py`: YAML/JSON manifest、相对路径、字段缺失错误。
- `tests/test_evaluation.py`: 成功率、失败原因、candidate count、best score、timings 聚合。
- `tests/test_benchmarking.py`: warmup、repeat、分位数、吞吐、失败样本处理。

测试不依赖真实 YOLO、RNG、GPU 或模型权重。使用 mock pipeline、mock backend 或可注入 pipeline builder。

## Phases

### Phase 1: Python Inference API

实现 `RGBDGrasp`、`predict_one()`、`predict()`、输入标准化、`info()`。

### Phase 2: Unlabeled Validation

实现 manifest 解析、`val()` 和工程指标聚合。

### Phase 3: Benchmarking

实现 warmup、repeat、阶段耗时统计、吞吐和失败率报告。

### Phase 4: CLI Reuse

让 `examples/run_image_pair.py` 复用 `RGBDGrasp.predict_one()`，减少重复装配逻辑，同时保持现有 CLI 参数和输出行为。
