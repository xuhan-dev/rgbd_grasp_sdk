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
