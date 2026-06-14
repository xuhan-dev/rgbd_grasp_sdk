#!/usr/bin/env bash
set -euo pipefail

MODE="dev"
CUDA="cu121"
RUN_TESTS="0"

usage() {
  cat <<'EOF'
Usage: scripts/setup_env.sh [options]

Options:
  --mode MODE     Install mode: base, dev, yolo, fastsam, rng, all. Default: dev
  --cuda CUDA     CUDA wheel profile for rng mode: cu121 or cpu. Default: cu121
  --run-tests     Run pytest -q after installation
  -h, --help      Show this help

Examples:
  scripts/setup_env.sh --mode dev
  scripts/setup_env.sh --mode yolo
  scripts/setup_env.sh --mode rng --cuda cu121
  scripts/setup_env.sh --mode all --cuda cu121 --run-tests
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:?--mode requires a value}"
      shift 2
      ;;
    --cuda)
      CUDA="${2:?--cuda requires a value}"
      shift 2
      ;;
    --run-tests)
      RUN_TESTS="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_python() {
  python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required")
print("python", sys.version.split()[0])
PY
}

install_base() {
  python3 -m pip install --upgrade pip setuptools wheel
  python3 -m pip install -e .
}

install_dev() {
  python3 -m pip install -e ".[dev]"
}

install_yolo() {
  python3 -m pip install -e ".[yolo]"
}

install_fastsam() {
  python3 -m pip install -e ".[fastsam]"
}

install_rng() {
  case "${CUDA}" in
    cu121)
      python3 -m pip install --force-reinstall \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1+cu121 torchvision==0.19.1+cu121
      python3 -m pip install iopath
      python3 -m pip install pytorch3d \
        -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
      ;;
    cpu)
      python3 -m pip install torch==2.4.1 torchvision==0.19.1
      python3 -m pip install iopath
      ;;
    *)
      echo "Unsupported --cuda value: ${CUDA}" >&2
      exit 2
      ;;
  esac
  python3 -m pip install -e ".[rng]"
}

require_python

case "${MODE}" in
  base)
    install_base
    ;;
  dev)
    install_base
    install_dev
    ;;
  yolo)
    install_base
    install_yolo
    ;;
  fastsam)
    install_base
    install_fastsam
    ;;
  rng)
    install_base
    install_rng
    ;;
  all)
    install_base
    install_dev
    install_yolo
    install_fastsam
    install_rng
    ;;
  *)
    echo "Unsupported --mode value: ${MODE}" >&2
    exit 2
    ;;
esac

if [[ "${RUN_TESTS}" == "1" ]]; then
  pytest -q
fi

echo "Environment setup completed: mode=${MODE}, cuda=${CUDA}"
