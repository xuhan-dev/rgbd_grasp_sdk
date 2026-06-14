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
scripts/setup_env.sh --mode rng --cuda cu121
```

`RegionNormalizedGrasp` 通过 `third_party/RegionNormalizedGrasp` 子模块接入。

真实 GPU smoke test 需要注意：

- PyTorch 轮子必须匹配本机 NVIDIA 驱动支持的 CUDA 版本。
- 当前已验证组合：`torch==2.4.1+cu121`、`torchvision==0.19.1+cu121`、`pytorch3d==0.7.8`、`grasp_nms==1.0.2`。
- RNG 配置中的相机内参路径需要放在 `grasping.model_config.intrinsics_path`，该目录下应包含 `camera_intrinsics.npz`。
- 旧 RNG 代码依赖真实 `pytorch3d` 和真实 `grasp_nms`。SDK 默认不启用轻量 fallback；只有显式配置 `allow_dependency_fallbacks: true` 时才会注册测试 fallback。

本地 smoke 命令示例：

```bash
scripts/smoke_real_gpu.sh
```

脚本默认使用 `data/smoke/smoke_yolo_rng.yaml`、`data/smoke/camera_intrinsics.npz` 和 `outputs/smoke/result.json`，这些路径可以通过环境变量覆盖。

默认不打开 3D 可视化窗口，避免自动化 smoke 被 GUI 阻塞。需要查看最终抓取结果时，可以在配置中设置：

```yaml
outputs:
  visualize_3d: true
```

3D 可视化中，全部预测夹爪使用灰色实体夹爪，最终选中的夹爪使用红色实体夹爪。

也可以在命令行临时覆盖：

```bash
python3 examples/run_image_pair.py ... --visualize-3d
python3 examples/run_image_pair.py ... --no-visualize-3d
```

## FastSAM

FastSAM 依赖需要按实际环境配置源码路径或安装包。基础 SDK 不在 import 阶段加载 FastSAM。
