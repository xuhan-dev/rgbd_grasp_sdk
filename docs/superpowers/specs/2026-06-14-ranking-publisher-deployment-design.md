# Ranking Publisher Deployment Design

## Goal

在不增加服务化入口、不裁剪 RNG 输入的前提下，继续增强目标抓取稳定性和外部系统本地对接能力。

## Scope

本轮优化包含三项：

1. 增强 `mask_aware` 抓取排序。
2. 让 CLI 使用 publisher 层，并增加本地 TF JSON 输出。
3. 补充部署文档。

不包含 HTTP、ZeroMQ、ROS2、DDS、长期运行服务入口，也不修改 RegionNormalizedGrasp 模型结构。

## Design

### Mask-Aware Ranking

RNG 继续对完整 RGB-D 场景生成抓取候选。排序阶段在现有 `center_in_mask` 基础上新增 `mask_overlap_ratio`，用抓取中心和夹爪宽度在图像平面近似构造候选区域，计算该区域与目标 mask 的重叠比例。

`target_score` 使用中心点命中和 overlap 的组合：

```text
target_score = max(center_score, mask_overlap_ratio)
```

`final_score` 继续使用配置权重组合：

```text
final_score = rng_score * w_rng + target_score * w_target
```

### CLI Publisher

`examples/run_image_pair.py` 不再手写 JSON 文件输出，改用 `JsonFilePublisher`。新增参数：

```bash
--output-transform-json outputs/grasp_tf.json
```

该参数只写本地 TF message JSON 文件，不启动服务，不绑定任何通信中间件。

### Deployment Docs

新增 `docs/deployment.md`，说明：

- Python/CUDA/PyTorch/PyTorch3D/grasp_nms 版本组合。
- 权重和 smoke 数据的推荐相对路径。
- 相机内参格式。
- 真实 GPU smoke 运行方式。
- 3D 可视化开关。

## Testing

- `tests/test_mask_aware_ranker.py` 覆盖 overlap 评分。
- `tests/test_cli_contract.py` 覆盖 CLI 参数和本地 TF JSON 输出。
- 全量运行 `pytest -q`。
- 运行 `scripts/smoke_real_gpu.sh`，真实 GPU smoke 应返回 `status: success`。
