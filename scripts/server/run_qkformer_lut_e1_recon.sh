#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu
if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-lut-e1] missing python3/python"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RESULT_TAG="${QKFORMER_LUT_RESULT_TAG:-}"
if [[ -n "$RESULT_TAG" ]]; then
  RESULT_TAG="_$(printf '%s' "$RESULT_TAG" | tr -cs 'A-Za-z0-9._-' '_')"
fi
RESULT_DIR="$ROOT/results/qkformer_lut_e1_recon_${TS}${RESULT_TAG}"
LOG_FILE="$RESULT_DIR/train_log.txt"
CONFIG_FILE="${QKFORMER_LUT_E1_CONFIG:-configs/qkformer_lut_e1_recon.yaml}"
mkdir -p "$RESULT_DIR"

if [[ -z "${QKFORMER_LUT_DATA_DIR:-}" && -d "$ROOT/data/cifar10/cifar-10-batches-py" ]]; then
  export QKFORMER_LUT_DATA_DIR="$ROOT/data/cifar10"
fi

if [[ -z "${QKFORMER_LUT_CKPT:-}" && -d "$ROOT/results" ]]; then
  export QKFORMER_LUT_CKPT="auto"
fi

if [[ "${QKFORMER_LUT_CKPT:-}" == "auto" || "${QKFORMER_LUT_CKPT:-}" == "latest" ]]; then
  QKFORMER_LUT_CKPT="$("$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
train_dirs = sorted(root.glob("results/qkformer_cifar10_train_*"))
if not train_dirs:
    raise SystemExit("no results/qkformer_cifar10_train_* directories found")
latest_dir = max(train_dirs, key=lambda path: (path.stat().st_mtime, str(path)))

manifest_path = latest_dir / "checkpoint_manifest.json"
checkpoint = ""
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = manifest.get("best_checkpoint") or manifest.get("latest_checkpoint") or ""

if not checkpoint:
    checkpoints = [
        path
        for pattern in ("*.pth", "*.pth.tar")
        for path in latest_dir.rglob(pattern)
    ]
    best = [path for path in checkpoints if "best" in path.name.lower()]
    candidates = best or checkpoints
    if candidates:
        checkpoint = str(max(candidates, key=lambda path: (path.stat().st_mtime, str(path))).resolve())

if not checkpoint:
    raise SystemExit(f"no checkpoint found under {latest_dir}")
path = Path(checkpoint).expanduser()
if not path.exists():
    raise SystemExit(f"selected checkpoint does not exist: {path}")
print(path)
PY
)"
  export QKFORMER_LUT_CKPT
fi

{
  echo "[qk-lut-e1] root=$ROOT"
  echo "[qk-lut-e1] result_dir=$RESULT_DIR"
  echo "[qk-lut-e1] start=$(date -Is)"
  echo "[qk-lut-e1] commit=$(git rev-parse --short HEAD)"
  echo "[qk-lut-e1] python=$PYTHON_BIN"
  echo "[qk-lut-e1] config=$CONFIG_FILE"
  echo "[qk-lut-e1] data_dir=${QKFORMER_LUT_DATA_DIR:-configs/qkformer_lut_e1_recon.yaml default}"
  echo "[qk-lut-e1] checkpoint=${QKFORMER_LUT_CKPT:-configs/qkformer_lut_e1_recon.yaml default}"
  echo "[qk-lut-e1] seed=${QKFORMER_LUT_E1_SEED:-config default}"
  echo "[qk-lut-e1] calib_batches=${QKFORMER_LUT_E1_CALIB_BATCHES:-config default}"
  echo "[qk-lut-e1] eval_batches=${QKFORMER_LUT_E1_EVAL_BATCHES:-config default}"
  echo "[qk-lut-e1] calib_shuffle=${QKFORMER_LUT_E1_CALIB_SHUFFLE:-config default}"

  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = ["torch", "yaml", "timm", "torchvision", "spikingjelly", "cupy"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, str(exc)))

if missing:
    for name, exc in missing:
        print(f"[qk-lut-e1] missing dependency: {name}: {exc}")
    print("[qk-lut-e1] run: bash scripts/server/install_qkformer_lut_deps.sh")
    sys.exit(1)

import torch
print(f"[qk-lut-e1] torch={torch.__version__}")
print(f"[qk-lut-e1] cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("[qk-lut-e1] CUDA is required because upstream QKFormer uses cupy-backed LIF nodes")
    sys.exit(1)
PY
  qk_lut_log_gpu "$PYTHON_BIN" "qk-lut-e1"

  "$PYTHON_BIN" tools/qkformer_lut_e1_recon.py \
    --config "$CONFIG_FILE" \
    --output-dir "$RESULT_DIR"

  "$PYTHON_BIN" - "$RESULT_DIR/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metrics = json.loads(path.read_text())
required = [
    "overall_reconstruction",
    "module_reconstruction",
    "per_stage_reconstruction",
    "calibration_prototypes",
]
missing = [key for key in required if key not in metrics]
if missing:
    raise SystemExit(f"metrics.json missing keys: {missing}")
overall = metrics["overall_reconstruction"]
print(f"[qk-lut-e1] metrics_ok={path}")
print(f"[qk-lut-e1] verdict={metrics.get('verdict')}")
print(f"[qk-lut-e1] address_lut_mse={overall.get('address_lut_mse')}")
print(f"[qk-lut-e1] address_relative_mse_reduction={overall.get('address_relative_mse_reduction')}")
print(f"[qk-lut-e1] token_channel_lut_mse={overall.get('token_channel_lut_mse')}")
print(f"[qk-lut-e1] shuffled_address_lut_mse={overall.get('shuffled_address_lut_mse')}")
PY

  echo "[qk-lut-e1] done=$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
