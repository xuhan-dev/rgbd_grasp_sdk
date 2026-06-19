# RGB-D Grasp SDK

模块化 RGB-D 目标分割与 6DoF 抓取预测 SDK。

本项目从 RGB-D 图像中识别目标物体，生成目标区域 mask，调用 RegionNormalizedGrasp 预测 6DoF 抓取候选，并输出最佳抓取结果、候选列表和本地 TF JSON。核心 pipeline 不绑定机械臂、相机驱动、DDS、ROS2、HTTP 或长期运行服务，便于后续替换分割模型、抓取模型、排序策略和外部发布方式。

## Features

- RGB-D 单帧输入：RGB、depth、相机内参、目标文本或类别。
- 可替换目标分割后端：YOLO、FastSAM 适配入口。
- 可替换抓取预测后端：RegionNormalizedGrasp 适配入口。
- mask-aware 抓取排序：`center_in_mask`、`mask_overlap_ratio`、`target_score`、`final_score`。
- 本地结果输出：
  - pipeline 结果摘要 JSON
  - 抓取 TF JSON
- 3D 可视化开关：目标 mask 区域在原始点云颜色上叠加青绿色蒙版，全部候选夹爪与最终夹爪使用不同颜色。
- GraspNet-1Billion 演示数据整理工具。
- 真实 GPU smoke 脚本。

## Non-Goals

当前项目不包含：

- 机械臂控制。
- 相机采集驱动。
- DDS/FastDDS。
- ROS2、HTTP、ZeroMQ 服务入口。
- 权重或公开数据集文件入库。

## Repository Layout

```text
src/rgbd_grasp_sdk/
  segmentation/      目标分割适配器
  grasping/          RNG 抓取预测适配器
  filtering/         mask 与候选过滤
  ranking/           mask-aware 排序
  pipeline/          主流程编排
  publishers/        本地 JSON / TF 输出
  visualization/     3D 可视化
  serialization/     结果序列化
  config/            YAML 配置加载
  compat/            第三方兼容层

examples/            CLI 示例
configs/             示例配置
scripts/             环境、smoke、数据整理脚本
docs/                项目文档
third_party/         第三方源码或子模块
```

## Quick Start

基础开发环境：

```bash
scripts/setup_env.sh --mode dev
pytest -q
```

YOLO 分割环境：

```bash
scripts/setup_env.sh --mode yolo
```

真实 RNG/GPU 环境：

```bash
scripts/setup_env.sh --mode rng --cuda cu121
```

完整模型环境：

```bash
scripts/setup_env.sh --mode all --cuda cu121 --run-tests
```

脚本不会下载模型权重或公开数据集。权重和数据请放在 `data/`、`third_party/` 或配置文件指定路径中。

运行时依赖检查：

```bash
scripts/check_runtime_deps.py --profile base
scripts/check_runtime_deps.py --profile rng-cu121 --rng-checkpoint third_party/RegionNormalizedGrasp/checkpoints/realsense
```

## Single-Frame CLI

```bash
python examples/run_image_pair.py \
  --config configs/yolo_rng.yaml \
  --rgb data/rgb.png \
  --depth data/depth.png \
  --intrinsics data/camera_intrinsics.npz \
  --target apple \
  --output-json outputs/result.json \
  --output-transform-json outputs/grasp_tf.json \
  --no-visualize-3d
```

输出：

- `outputs/result.json`: pipeline 结果摘要 JSON。
- `outputs/grasp_tf.json`: 外部系统可消费的本地 TF message。

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

For batch experiments, pass explicit sample lists to `predict()`. `val()` and `benchmark()` accept explicit sample lists or manifest files.

## Real GPU Smoke

真实依赖、真实权重和 GPU 环境准备好后运行：

```bash
scripts/smoke_real_gpu.sh
```

预期输出包含：

```text
cuda_available True
status: success
best_score:
best_center_px:
```

## Demo Data

GraspNet-1Billion 原始数据下载后，可以抽取单帧为项目统一输入格式：

```bash
scripts/prepare_graspnet_demo.py \
  --root data/raw/graspnet \
  --output-dir data/demo/graspnet_scene_0000_frame_0000 \
  --scene-id 0 \
  --frame-id 0 \
  --camera realsense \
  --target apple
```

详见 `docs/demo_data.md`。

## Documentation

- `docs/installation.md`: 环境安装与一键配置脚本。
- `docs/architecture.md`: 架构和模块边界。
- `docs/model_adapters.md`: YOLO、FastSAM、RegionNormalizedGrasp 适配说明。
- `docs/deployment.md`: 真实环境部署说明。
- `docs/demo_data.md`: GraspNet 演示数据准备。
- `docs/smoke_tests.md`: 单元测试和真实 GPU smoke。
- `docs/transform_contract.md`: 抓取 TF 输出合约。
- `THIRD_PARTY_LICENSES.md`: 第三方依赖、第三方源码和数据许可证清单。
- `MODEL_WEIGHTS.md`: 模型权重和公开数据使用说明。

## License

主项目代码使用 Apache-2.0 许可证，详见 `LICENSE`。

第三方源码、Python 依赖、模型权重和公开数据集按各自许可证或数据条款使用。发布 wheel、Docker 镜像、模型包或演示数据前，请先核对 `THIRD_PARTY_LICENSES.md` 和 `MODEL_WEIGHTS.md`。

## Development

```bash
scripts/setup_env.sh --mode dev
pytest -q
```

提交前建议运行：

```bash
pytest -q
git diff --check
```
