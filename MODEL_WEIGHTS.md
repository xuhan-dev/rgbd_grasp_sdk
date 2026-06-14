# Model Weights and Data Policy

本项目仓库不包含模型权重、公开数据集原始文件或需要单独授权的大型资产。

## Supported Layout

推荐把本地权重和数据放在以下路径，并通过配置文件或 CLI 参数引用：

```text
data/
  weights/
  raw/
  demo/

third_party/
  RegionNormalizedGrasp/
    checkpoints/
```

这些路径用于本地运行和 smoke test，不代表相关文件可随主项目一起发布。

## Required Metadata

每个外部权重目录建议附带一个 `metadata.yaml`，记录：

```yaml
name: RegionNormalizedGrasp realsense checkpoint
source: upstream project or download page
version: unknown
license: to be verified
usage: research/demo only until verified
sha256: optional-checksum
downloaded_at: YYYY-MM-DD
notes: >
  Record any upstream usage restriction, citation requirement, or redistribution limit.
```

## RegionNormalizedGrasp Checkpoints

RNG checkpoint 是抓取预测的核心资产。当前项目只提供适配器和配置示例，不声明 checkpoint 的授权范围。

使用规则：

- 本地 smoke test 可以通过 `--rng-checkpoint` 或配置文件指定 checkpoint 路径。
- 不要把 checkpoint 提交到 git。
- 不要在未确认上游许可证前把 checkpoint 打进 wheel、Docker 镜像或发布包。
- 对外演示时保留 checkpoint 来源、版本和引用论文信息。

## YOLO and FastSAM Weights

YOLO/FastSAM 权重属于可选分割后端资产。

使用规则：

- 通过配置文件指定权重路径。
- 确认具体权重文件对应的上游许可证，而不是只看 Python 包许可证。
- `ultralytics` 相关模型在闭源或商业场景中需要特别评估 AGPL/commercial 授权。
- 示例配置中的路径只是占位，不表示项目分发这些权重。

## GraspNet Data

`scripts/prepare_graspnet_demo.py` 可从用户本地 GraspNet-1Billion 原始数据中抽取单帧演示数据。

使用规则：

- 项目脚本不下载 GraspNet 原始数据。
- 抽取后的样例仍受 GraspNet 数据集条款约束。
- 如果要把样例图像、深度图或内参文件随项目发布，先确认数据集条款允许再分发，并在 `THIRD_PARTY_LICENSES.md` 中记录来源。

## Git Hygiene

建议 `.gitignore` 保持以下资产不入库：

```text
data/raw/
data/weights/
third_party/RegionNormalizedGrasp/checkpoints/
*.pt
*.pth
*.ckpt
*.onnx
```

若确实需要发布小型演示资产，应记录来源、许可证和校验和，并避免混入真实生产数据。
