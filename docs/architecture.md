# 架构说明

## 核心原则

- `pipeline` 只依赖抽象接口，不依赖具体模型。
- 真实模型接入必须放在 adapter 内。
- SDK 对外只返回自有数据结构。
- 可选 backend 未安装时，基础包仍可 import。

## 数据流

```text
RGB-D 输入
  -> Segmenter
  -> MaskPostProcessor
  -> GraspPredictor
  -> GraspFilter
  -> GraspRanker
  -> PipelineResult
```

## 后续模型接入方式

新增分割模型：

1. 在 `src/rgbd_grasp_sdk/segmentation/` 新增 adapter。
2. adapter 实现 `Segmenter` 协议。
3. 在 `segmentation/factory.py` 注册 backend 名称。
4. 新增不依赖 GPU 的 contract test。

新增抓取模型：

1. 在 `src/rgbd_grasp_sdk/grasping/` 新增 adapter。
2. adapter 实现 `GraspPredictor` 协议。
3. 在 `grasping/factory.py` 注册 backend 名称。
4. 将模型原生输出转换为 `GraspCandidate`。
