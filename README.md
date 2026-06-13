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
  --target apple
```

第一阶段的真实 YOLO/FastSAM/RNG adapter 尚未接入。CLI 主要用于验证配置、输入读取和 pipeline 组合方式。
