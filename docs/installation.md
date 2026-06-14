# Installation

本文档说明如何配置 RGB-D Grasp SDK 的开发、分割模型和真实 RNG/GPU 环境。

## Requirements

- Python 3.10+
- pip
- Linux
- 真实 GPU 推理建议使用 NVIDIA GPU 和 CUDA 12.1 对应 PyTorch 轮子

## One-Command Setup

项目提供一键环境配置脚本：

```bash
scripts/setup_env.sh --mode dev
```

脚本不会下载模型权重或公开数据集，只安装 Python 依赖。

许可证治理入口：

- 主项目许可证：`LICENSE`
- 第三方依赖清单：`THIRD_PARTY_LICENSES.md`
- 模型权重和数据说明：`MODEL_WEIGHTS.md`

依赖分层文件位于：

```text
requirements/
  base.txt
  dev.txt
  yolo.txt
  fastsam.txt
  rng-common.txt
  rng-cu121.txt
  rng-cpu.txt
```

`scripts/setup_env.sh` 会读取这些文件安装依赖，避免把复杂 CUDA/PyTorch3D 细节写死在脚本逻辑中。

## Modes

基础运行依赖：

```bash
scripts/setup_env.sh --mode base
```

开发依赖：

```bash
scripts/setup_env.sh --mode dev
```

YOLO 分割依赖：

```bash
scripts/setup_env.sh --mode yolo
```

FastSAM 依赖：

```bash
scripts/setup_env.sh --mode fastsam
```

RegionNormalizedGrasp 真实 GPU 依赖：

```bash
scripts/setup_env.sh --mode rng --cuda cu121
```

完整依赖：

```bash
scripts/setup_env.sh --mode all --cuda cu121
```

安装后运行测试：

```bash
scripts/setup_env.sh --mode all --cuda cu121 --run-tests
```

## CUDA Profiles

当前脚本支持：

- `--cuda cu121`: 安装 `torch==2.4.1+cu121`、`torchvision==0.19.1+cu121` 和对应 PyTorch3D wheel。
- `--cuda cpu`: 安装 CPU PyTorch，不安装 PyTorch3D CUDA wheel。

真实 RNG smoke 已验证组合：

```text
torch 2.4.1+cu121
torchvision 0.19.1+cu121
pytorch3d 0.7.8
grasp_nms 1.0.2
```

## Manual Setup

基础开发：

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev]"
pytest -q
```

YOLO：

```bash
python3 -m pip install -e ".[yolo]"
```

RNG/GPU：

```bash
python3 -m pip install --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1+cu121 torchvision==0.19.1+cu121
python3 -m pip install iopath
python3 -m pip install pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
python3 -m pip install -e ".[rng]"
```

## Weights And Data

权重和数据不进入版本库。推荐路径：

```text
data/
  weights/
  raw/
  demo/
outputs/
third_party/
  RegionNormalizedGrasp/
```

`data/` 和 `outputs/` 已被 `.gitignore` 忽略。

## Verification

基础验证：

```bash
pytest -q
```

运行时依赖检查：

```bash
scripts/check_runtime_deps.py --profile base
scripts/check_runtime_deps.py --profile yolo --yolo-weights data/weights/yolo11x-seg.pt
scripts/check_runtime_deps.py \
  --profile rng-cu121 \
  --rng-checkpoint third_party/RegionNormalizedGrasp/checkpoints/realsense \
  --intrinsics data/smoke/camera_intrinsics.npz
```

真实 GPU smoke：

```bash
scripts/smoke_real_gpu.sh
```

如果真实权重或数据路径不同，通过环境变量覆盖，详见 `docs/smoke_tests.md`。
