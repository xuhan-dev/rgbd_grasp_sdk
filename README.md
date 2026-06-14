# RGB-D Grasp SDK

一个模块化 RGB-D 目标分割和 6DoF 抓取预测 SDK。

第一阶段目标：

- 输入 RGB、depth、相机内参和目标描述。
- 通过可替换 `Segmenter` 生成目标 mask。
- 通过可替换 `GraspPredictor` 生成抓取候选。
- 过滤、排序并返回结构化结果。

第一阶段不包含：

- 机械臂控制。
- 相机控制。
- DDS/FastDDS。
- ROS2/HTTP/ZeroMQ 发布。

## 开发环境

```bash
conda env create -f environment.yml
conda activate rgbd-grasp-sdk
pip install -e ".[dev]"
pytest -q
```

## 单帧 CLI

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

CLI 可用于验证配置、输入读取、pipeline 组合方式和 JSON 输出；真实模型运行仍依赖相应 extras、权重和运行环境。

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

## Smoke Tests

详见 `docs/smoke_tests.md`。
