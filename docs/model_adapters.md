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

真实 GPU smoke test 需要注意：

- PyTorch 轮子必须匹配本机 NVIDIA 驱动支持的 CUDA 版本。
- RNG 配置中的相机内参路径需要放在 `grasping.model_config.intrinsics_path`，该目录下应包含 `camera_intrinsics.npz`。
- 旧 RNG 代码还会依赖 `pytorch3d` 和 `grasp_nms`。如果当前 Python/CUDA 组合没有可安装轮子，可以用临时 fallback 先验证真实权重前向链路；生产环境应安装或编译对应实现。

本地 smoke 命令示例：

```bash
PYTHONPATH=/tmp/rgbd_grasp_smoke/site:../sam_rng:../sam_rng/RegionNormalizedGrasp \
python3 examples/run_image_pair.py \
  --config /tmp/rgbd_grasp_smoke/smoke_yolo_rng.yaml \
  --rgb third_party/RegionNormalizedGrasp/images/demo_rgb.png \
  --depth third_party/RegionNormalizedGrasp/images/demo_depth.png \
  --intrinsics /tmp/rgbd_grasp_smoke/camera_intrinsics.npz \
  --target microwave \
  --output-json /tmp/rgbd_grasp_smoke/result.json
```

## FastSAM

FastSAM 依赖需要按实际环境配置源码路径或安装包。基础 SDK 不在 import 阶段加载 FastSAM。
