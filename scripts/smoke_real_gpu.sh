#!/usr/bin/env bash
set -euo pipefail

SMOKE_CONFIG="${SMOKE_CONFIG:-data/smoke/smoke_yolo_rng.yaml}"
SMOKE_RGB="${SMOKE_RGB:-third_party/RegionNormalizedGrasp/images/demo_rgb.png}"
SMOKE_DEPTH="${SMOKE_DEPTH:-third_party/RegionNormalizedGrasp/images/demo_depth.png}"
SMOKE_INTRINSICS="${SMOKE_INTRINSICS:-data/smoke/camera_intrinsics.npz}"
SMOKE_TARGET="${SMOKE_TARGET:-microwave}"
SMOKE_OUTPUT_JSON="${SMOKE_OUTPUT_JSON:-outputs/smoke/result.json}"
SMOKE_PYTHONPATH="${SMOKE_PYTHONPATH:-../sam_rng:../sam_rng/RegionNormalizedGrasp}"

python3 - <<'PY'
import torch
import pytorch3d
import grasp_nms

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("pytorch3d", getattr(pytorch3d, "__version__", "unknown"))
print("grasp_nms", getattr(grasp_nms, "__version__", "unknown"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY

PYTHONPATH="${SMOKE_PYTHONPATH}" \
python3 examples/run_image_pair.py \
  --config "${SMOKE_CONFIG}" \
  --rgb "${SMOKE_RGB}" \
  --depth "${SMOKE_DEPTH}" \
  --intrinsics "${SMOKE_INTRINSICS}" \
  --target "${SMOKE_TARGET}" \
  --output-json "${SMOKE_OUTPUT_JSON}" \
  --no-visualize-3d
