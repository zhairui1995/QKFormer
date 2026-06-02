#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "[qk-lut-e0] missing python3/python"
    exit 1
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$ROOT/results/qkformer_lut_e0_diag_${TS}"
LOG_FILE="$RESULT_DIR/train_log.txt"
mkdir -p "$RESULT_DIR"

if [[ -z "${QKFORMER_LUT_DATA_DIR:-}" && -d "$ROOT/data/cifar10/cifar-10-batches-py" ]]; then
  export QKFORMER_LUT_DATA_DIR="$ROOT/data/cifar10"
fi

{
  echo "[qk-lut-e0] root=$ROOT"
  echo "[qk-lut-e0] result_dir=$RESULT_DIR"
  echo "[qk-lut-e0] start=$(date -Is)"
  echo "[qk-lut-e0] commit=$(git rev-parse --short HEAD)"
  echo "[qk-lut-e0] python=$PYTHON_BIN"
  echo "[qk-lut-e0] data_dir=${QKFORMER_LUT_DATA_DIR:-configs/qkformer_lut_e0_diag.yaml default}"

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
        print(f"[qk-lut-e0] missing dependency: {name}: {exc}")
    print("[qk-lut-e0] run: bash scripts/server/install_qkformer_lut_deps.sh")
    sys.exit(1)

import torch
print(f"[qk-lut-e0] torch={torch.__version__}")
print(f"[qk-lut-e0] cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("[qk-lut-e0] CUDA is required because upstream QKFormer uses cupy-backed LIF nodes")
    sys.exit(1)
PY

  "$PYTHON_BIN" tools/qkformer_lut_e0_diag.py \
    --config configs/qkformer_lut_e0_diag.yaml \
    --output-dir "$RESULT_DIR"

  "$PYTHON_BIN" - "$RESULT_DIR/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metrics = json.loads(path.read_text())
required = [
    "address_coverage",
    "bucket_occupancy",
    "singleton_fraction",
    "conditional_variance",
    "candidate_background_variance",
    "per_stage_summary",
    "per_block_summary",
    "module_summary",
]
missing = [key for key in required if key not in metrics]
if missing:
    raise SystemExit(f"metrics.json missing keys: {missing}")
print(f"[qk-lut-e0] metrics_ok={path}")
print(f"[qk-lut-e0] verdict={metrics.get('verdict')}")
PY

  echo "[qk-lut-e0] done=$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
