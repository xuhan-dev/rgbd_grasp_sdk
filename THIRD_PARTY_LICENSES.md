# Third-Party Licenses

本文件用于维护 RGB-D Grasp SDK 的第三方许可证清单。主项目代码按 `LICENSE` 中的 Apache-2.0 授权；第三方源码、模型权重、公开数据集和外部 Python 包不自动继承主项目许可证。

发布二进制包、Docker 镜像、模型包或商业交付前，必须重新核对本清单和实际分发内容。

## Runtime Python Packages

| Package | Usage | Declared license | Distribution note |
| --- | --- | --- | --- |
| `numpy` | 数组计算 | BSD-3-Clause | 通过 pip/conda 安装，不随源码入库。 |
| `opencv-python` | 图像读写与处理 | Apache-2.0 | 通过 pip 安装，不随源码入库。 |
| `PyYAML` | YAML 配置 | MIT | 通过 pip 安装，不随源码入库。 |
| `pytest` | 测试 | MIT | 仅开发环境依赖。 |
| `torch` | 模型运行 | BSD-style | 通过 PyTorch 官方 wheel 安装，不随源码入库。 |
| `torchvision` | 模型运行 | BSD-style | 通过 PyTorch 官方 wheel 安装，不随源码入库。 |
| `pytorch3d` | RNG 几何算子 | BSD-style | 通过官方 wheel 安装，不随源码入库。 |
| `ultralytics` | 可选 YOLO 分割后端 | AGPL-3.0 / commercial options | 仅可选依赖；商业或闭源分发需单独评估。 |
| `open3d` | 点云处理/可视化 | MIT | 通过 pip 安装，不随源码入库。 |
| `transforms3d` | 位姿变换 | BSD | 通过 pip 安装，不随源码入库。 |
| `scikit-image` | 图像处理 | BSD-3-Clause | 通过 pip 安装，不随源码入库。 |
| `trimesh` | 几何处理 | MIT | 通过 pip 安装，不随源码入库。 |
| `numba` | 加速计算 | BSD-2-Clause | 通过 pip 安装，不随源码入库。 |
| `cupoch` | GPU 点云/几何依赖 | MIT | 通过 pip 安装，不随源码入库。 |
| `autolab_core` | RNG 上游依赖 | Berkeley research/non-commercial terms in upstream components | 需按实际安装包和依赖链复核。 |
| `cvxopt` | 优化依赖 | GPL-3.0-or-later | 通过 pip 安装；分发形态需单独评估 GPL 影响。 |
| `matplotlib` | 可视化/调试 | PSF-compatible | 通过 pip 安装，不随源码入库。 |
| `iopath` | PyTorch3D 依赖 | MIT | 通过 pip 安装，不随源码入库。 |
| `grasp_nms` | RNG NMS 扩展 | To be verified | 当前项目只声明依赖；发布前需确认 PyPI/源码许可证。 |

## Vendored or Submodule Code

| Path | Source | License status | Action |
| --- | --- | --- | --- |
| `third_party/RegionNormalizedGrasp/` | RegionNormalizedGrasp upstream code | No top-level license file found in the vendored checkout | 发布前必须向上游仓库或作者确认可分发许可证；不能默认按本项目 Apache-2.0 授权。 |
| `third_party/RegionNormalizedGrasp/customgraspnetAPI/utils/dexnet/LICENSE` | Dex-Net related utility code | UC Berkeley educational, research, and not-for-profit permission; commercial licensing requires separate contact | 商业用途或再分发前必须单独评估。 |
| `third_party/grasp_nms/` | 本项目的本地说明占位目录 | Project documentation only | 真正的 `grasp_nms` 扩展来自 pip/源码安装，许可证见上表待确认项。 |

## Model Backends

| Backend | Code status | Weights status | License note |
| --- | --- | --- | --- |
| YOLO via `ultralytics` | 可选 Python 依赖 | 不入库 | AGPL/commercial 双轨风险较高；闭源或商业部署需使用合规授权。 |
| FastSAM adapter | 适配入口，不入库上游源码 | 不入库 | 使用具体 FastSAM 权重/源码前需确认其上游许可证。 |
| RegionNormalizedGrasp | 子模块/第三方源码 | checkpoint 不应随主项目发布，除非许可证明确允许 | 当前 checkout 未发现顶层 LICENSE，发布前为阻断项。 |

## Data Assets

| Asset | Status | License note |
| --- | --- | --- |
| `data/demo/graspnet_scene_0000_frame_0000/` | 本地演示数据目录 | 若来源于 GraspNet-1Billion，仅用于演示/验证；再分发需遵守 GraspNet 数据集条款。 |
| GraspNet-1Billion raw data | 不入库 | 用户自行下载，项目脚本只做格式整理。 |

## Release Checklist

- 确认 `third_party/RegionNormalizedGrasp/` 的上游许可证和可再分发范围。
- 确认 `grasp_nms` 的源码许可证和二进制扩展分发条款。
- 若发布 Docker 镜像，生成镜像内 Python 包许可证清单。
- 若发布模型权重，逐个记录来源、版本、下载地址、许可证、允许用途和校验和。
- 若发布 GraspNet 或其他公开数据样例，确认数据集条款允许再分发。
- 对商业/闭源部署，重点复核 `ultralytics`、`cvxopt`、Dex-Net 相关代码和 RNG 权重。
