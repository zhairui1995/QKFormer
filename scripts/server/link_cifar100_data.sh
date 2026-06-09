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
    echo "[qk-lut-c100-data] missing python3/python"
    exit 1
  fi
fi

SOURCE_ROOT="${CIFAR100_SOURCE_ROOT:-/home/lbz/git-hub/datasets}"
TARGET_ROOT="${QKFORMER_LUT_CIFAR100_DATA_DIR:-$ROOT/data/cifar100}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$ROOT/results/qkformer_lut_cifar100_data_link_${TS}"
LOG_FILE="$RESULT_DIR/link_log.txt"
mkdir -p "$RESULT_DIR"

{
  echo "[qk-lut-c100-data] root=$ROOT"
  echo "[qk-lut-c100-data] source_root=$SOURCE_ROOT"
  echo "[qk-lut-c100-data] target_root=$TARGET_ROOT"
  mkdir -p "$TARGET_ROOT"

  if [[ -d "$SOURCE_ROOT/cifar-100-python" ]]; then
    ln -sfn "$SOURCE_ROOT/cifar-100-python" "$TARGET_ROOT/cifar-100-python"
    echo "[qk-lut-c100-data] linked cifar-100-python"
  else
    echo "[qk-lut-c100-data] missing source CIFAR-100 directory under $SOURCE_ROOT"
    exit 1
  fi

  if [[ -f "$SOURCE_ROOT/cifar-100-python.tar.gz" ]]; then
    ln -sfn "$SOURCE_ROOT/cifar-100-python.tar.gz" "$TARGET_ROOT/cifar-100-python.tar.gz"
    echo "[qk-lut-c100-data] linked cifar-100-python.tar.gz"
  fi

  export QKFORMER_LUT_CIFAR100_DATA_DIR="$TARGET_ROOT"
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["QKFORMER_LUT_CIFAR100_DATA_DIR"])
print(f"[qk-lut-c100-data] verify_root={root}")
print(f"[qk-lut-c100-data] batches_dir_exists={(root / 'cifar-100-python').is_dir()}")

try:
    from torchvision import datasets
    train_set = datasets.CIFAR100(root=str(root), train=True, download=False)
    test_set = datasets.CIFAR100(root=str(root), train=False, download=False)
    print(f"[qk-lut-c100-data] torchvision_train={len(train_set)}")
    print(f"[qk-lut-c100-data] torchvision_test={len(test_set)}")
except Exception as exc:
    print(f"[qk-lut-c100-data] torchvision_verify_failed={exc}")
    raise
PY

  echo "[qk-lut-c100-data] done=$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
