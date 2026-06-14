# Smoke Tests

## Unit Tests

```bash
scripts/setup_env.sh --mode dev
pytest -q
```

## Real GPU Smoke

真实 smoke 用于确认以下依赖可用：

- CUDA
- PyTorch
- PyTorch3D
- grasp_nms
- RegionNormalizedGrasp 权重
- 分割模型权重

运行：

```bash
scripts/smoke_real_gpu.sh
```

默认输入：

- `SMOKE_CONFIG`: `data/smoke/smoke_yolo_rng.yaml`
- `SMOKE_RGB`: `third_party/RegionNormalizedGrasp/images/demo_rgb.png`
- `SMOKE_DEPTH`: `third_party/RegionNormalizedGrasp/images/demo_depth.png`
- `SMOKE_INTRINSICS`: `data/smoke/camera_intrinsics.npz`
- `SMOKE_TARGET`: `microwave`
- `SMOKE_OUTPUT_JSON`: `outputs/smoke/result.json`
- `SMOKE_PYTHONPATH`: `../sam_rng:../sam_rng/RegionNormalizedGrasp`

这些变量都可以在命令行覆盖：

```bash
SMOKE_TARGET=apple scripts/smoke_real_gpu.sh
```

预期输出包含：

```text
status: success
best_score:
best_center_px:
```

该 smoke 默认使用 `--no-visualize-3d`，避免 Open3D GUI 阻塞自动化检查。

## Local Transform JSON

单帧 CLI 可以同时输出完整 pipeline 结果和抓取 TF message：

```bash
python examples/run_image_pair.py \
  --config configs/yolo_rng.yaml \
  --rgb data/rgb.png \
  --depth data/depth.png \
  --intrinsics data/camera_intrinsics.npz \
  --target apple \
  --output-json outputs/result.json \
  --output-transform-json outputs/grasp_tf.json
```
