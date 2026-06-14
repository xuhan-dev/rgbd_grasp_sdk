# Demo Data

正式演示推荐使用 GraspNet-1Billion 中抽取的少量 RGB-D 帧。项目不提交公开数据集原始文件，也不在脚本中自动下载数据；用户需要先按 GraspNet 官方说明下载数据到本地。

GraspNet 官网：

```text
https://www.graspnet.net/
```

官方数据下载页：

```text
https://graspnet.net/datasets.html
```

原始 RGB-D 场景以 zip 分包提供。常见分包包括：

- `train_1.zip`
- `train_2.zip`
- `train_3.zip`
- `train_4.zip`
- `test_seen.zip`
- `test_similar.zip`
- `test_novel.zip`

这些文件体积较大，Google Drive 链接可能触发下载限流。若命令行下载失败，请通过浏览器或网盘客户端下载，并把 zip 放到：

```text
data/raw/graspnet_downloads/
```

解压后确保根目录包含：

```text
data/raw/graspnet/
  scenes/
```

## Expected Input Layout

抽帧脚本支持常见 GraspNet 场景目录：

```text
<graspnet-root>/
  scenes/
    scene_0000/
      realsense/
        rgb/
          0000.png
        depth/
          0000.png
        meta/
          0000.npz
```

脚本支持两种内参来源。

`meta/0000.npz` 中包含以下任一 3x3 内参矩阵字段：

- `intrinsic_matrix`
- `K`
- `camera_intrinsic`

或官方原始目录中的相机级内参：

```text
<graspnet-root>/scenes/scene_0000/realsense/camK.npy
```

## Prepare One Demo Frame

```bash
scripts/prepare_graspnet_demo.py \
  --root data/raw/graspnet \
  --output-dir data/demo/graspnet_scene_0000_frame_0000 \
  --scene-id 0 \
  --frame-id 0 \
  --camera realsense \
  --target apple
```

如果使用 `train_4.zip`，场景编号通常不是从 0 开始。可以先查看可用场景：

```bash
find data/raw/graspnet/scenes -maxdepth 1 -type d | sort | head
```

然后把 `--scene-id` 改成实际存在的编号。

输出目录：

```text
data/demo/graspnet_scene_0000_frame_0000/
  rgb.png
  depth.png
  camera_intrinsics.npz
  metadata.yaml
```

## Run Demo

```bash
python examples/run_image_pair.py \
  --config configs/yolo_rng.yaml \
  --rgb data/demo/graspnet_scene_0000_frame_0000/rgb.png \
  --depth data/demo/graspnet_scene_0000_frame_0000/depth.png \
  --intrinsics data/demo/graspnet_scene_0000_frame_0000/camera_intrinsics.npz \
  --target apple \
  --output-json outputs/demo/result.json \
  --output-transform-json outputs/demo/grasp_tf.json \
  --no-visualize-3d
```

## Notes

- `target` 需要与当前分割模型的类别或文本能力匹配。
- 如果使用 YOLO segmentation 权重，建议选择 COCO 或自训练权重能识别的目标类别。
- GraspNet 原始标注可用于更完整评估，但本脚本只整理单帧 RGB-D 演示输入。
