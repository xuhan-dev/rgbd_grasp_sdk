# Deployment

本文档说明本项目在真实模型环境中的推荐部署方式。示例路径均为项目相对路径或可配置路径。

## Environment

当前已验证组合：

- Python 3.10
- CUDA 12.1 轮子组合
- `torch==2.4.1+cu121`
- `torchvision==0.19.1+cu121`
- `pytorch3d==0.7.8`
- `grasp_nms==1.0.2`

基础安装：

```bash
scripts/setup_env.sh --mode all --cuda cu121
```

真实 RNG 环境需要安装匹配的 PyTorch、PyTorch3D 和 `grasp_nms`。安装命令见 `docs/model_adapters.md`。
更完整的环境说明见 `docs/installation.md`。

## License Checks

部署前需要确认实际分发内容的许可证边界：

- 主项目代码按 `LICENSE` 中的 Apache-2.0 使用。
- Python 包、第三方源码和数据资产见 `THIRD_PARTY_LICENSES.md`。
- 模型权重和 GraspNet 数据样例见 `MODEL_WEIGHTS.md`。
- Docker 镜像、离线安装包和模型包都属于再分发形态，需要额外复核镜像内依赖、权重和数据条款。

部署前可检查运行时依赖：

```bash
scripts/check_runtime_deps.py \
  --profile rng-cu121 \
  --rng-checkpoint third_party/RegionNormalizedGrasp/checkpoints/realsense \
  --intrinsics data/smoke/camera_intrinsics.npz
```

## Third-Party Layout

推荐 third-party 目录：

```text
third_party/
  RegionNormalizedGrasp/
  grasp_nms/
```

RNG adapter 通过 `PYTHONPATH` 找到旧项目中的 `preprocessor/RNG.py`。真实 smoke 默认使用：

```bash
SMOKE_PYTHONPATH=../sam_rng:../sam_rng/RegionNormalizedGrasp
```

如旧项目路径不同，请通过环境变量覆盖。

## Weights

推荐权重位置：

```text
data/weights/
  yolo11x-seg.pt
third_party/RegionNormalizedGrasp/checkpoints/realsense/
```

配置示例：

```yaml
segmentation:
  backend: yolo
  model_path: data/weights/yolo11x-seg.pt

grasping:
  backend: rng
  checkpoint_path: third_party/RegionNormalizedGrasp/checkpoints/realsense
  device: cuda
  input_size: [360, 640]
  model_config:
    intrinsics_path: data/smoke
```

## Camera Intrinsics

相机内参使用 `.npz` 文件，包含矩阵 `K`。推荐路径：

```text
data/smoke/camera_intrinsics.npz
```

`K` 格式：

```text
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

## Single-Frame Run

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

- `outputs/result.json`: 完整 pipeline 结果。
- `outputs/grasp_tf.json`: 外部系统可消费的本地 TF message。

如需使用 GraspNet-1Billion 抽帧作为正式演示数据，详见 `docs/demo_data.md`。

## Real GPU Smoke

```bash
scripts/smoke_real_gpu.sh
```

默认输入：

```text
data/smoke/smoke_yolo_rng.yaml
data/smoke/camera_intrinsics.npz
third_party/RegionNormalizedGrasp/images/demo_rgb.png
third_party/RegionNormalizedGrasp/images/demo_depth.png
```

如果 smoke 资源放在其他位置，可以覆盖环境变量：

```bash
SMOKE_CONFIG=data/smoke/smoke_yolo_rng.yaml \
SMOKE_INTRINSICS=data/smoke/camera_intrinsics.npz \
SMOKE_OUTPUT_JSON=outputs/smoke/result.json \
scripts/smoke_real_gpu.sh
```

预期输出包含：

```text
cuda_available True
status: success
best_score:
best_center_px:
```

## 3D Visualization

配置文件：

```yaml
outputs:
  visualize_3d: true
```

命令行覆盖：

```bash
python examples/run_image_pair.py ... --visualize-3d
python examples/run_image_pair.py ... --no-visualize-3d
```

可视化约定：

- 目标 mask 区域：保留原始点云颜色，并叠加青绿色蒙版。
- 其他点云区域：保留原始点云颜色。
- 全部候选夹爪：灰色实体夹爪。
- 最终选中夹爪：红色实体夹爪。

## Notes

- RNG 输入保持完整 RGB-D 场景，不按目标 mask 裁剪。
- 目标 mask 只用于候选过滤和 mask-aware 排序。
- 本项目当前只提供本地 CLI 与本地文件输出，不包含 HTTP、ZeroMQ、ROS2、DDS 或长期运行服务入口。
