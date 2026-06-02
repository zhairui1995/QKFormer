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
    echo "[qk-lut-data] missing python3/python"
    exit 1
  fi
fi

SOURCE_ROOT="${CIFAR10_SOURCE_ROOT:-/home/lbz/git-hub/datasets}"
TARGET_ROOT="${QKFORMER_LUT_DATA_DIR:-$ROOT/data/cifar10}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$ROOT/results/qkformer_lut_data_link_${TS}"
LOG_FILE="$RESULT_DIR/link_log.txt"
mkdir -p "$RESULT_DIR"

{
  echo "[qk-lut-data] root=$ROOT"
  echo "[qk-lut-data] source_root=$SOURCE_ROOT"
  echo "[qk-lut-data] target_root=$TARGET_ROOT"
  mkdir -p "$TARGET_ROOT"

  if [[ -d "$SOURCE_ROOT/cifar-10-batches-py" ]]; then
    ln -sfn "$SOURCE_ROOT/cifar-10-batches-py" "$TARGET_ROOT/cifar-10-batches-py"
    echo "[qk-lut-data] linked cifar-10-batches-py"
  elif [[ -d "$SOURCE_ROOT/cifar10/cifar-10-batches-py" ]]; then
    ln -sfn "$SOURCE_ROOT/cifar10/cifar-10-batches-py" "$TARGET_ROOT/cifar-10-batches-py"
    echo "[qk-lut-data] linked cifar10/cifar-10-batches-py"
  else
    echo "[qk-lut-data] missing source CIFAR-10 directory under $SOURCE_ROOT"
    exit 1
  fi

  if [[ -f "$SOURCE_ROOT/cifar-10-python.tar.gz" ]]; then
    ln -sfn "$SOURCE_ROOT/cifar-10-python.tar.gz" "$TARGET_ROOT/cifar-10-python.tar.gz"
    echo "[qk-lut-data] linked cifar-10-python.tar.gz"
  fi

  export QKFORMER_LUT_DATA_DIR="$TARGET_ROOT"
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["QKFORMER_LUT_DATA_DIR"])
print(f"[qk-lut-data] verify_root={root}")
print(f"[qk-lut-data] batches_dir_exists={(root / 'cifar-10-batches-py').is_dir()}")

try:
    from torchvision import datasets
    train_set = datasets.CIFAR10(root=str(root), train=True, download=False)
    test_set = datasets.CIFAR10(root=str(root), train=False, download=False)
    print(f"[qk-lut-data] torchvision_train={len(train_set)}")
    print(f"[qk-lut-data] torchvision_test={len(test_set)}")
except Exception as exc:
    print(f"[qk-lut-data] torchvision_verify_failed={exc}")
    raise
PY

  echo "[qk-lut-data] done=$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
