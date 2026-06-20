#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-${QKFORMER_PYTHON:-python3}}"
DATA_DIR="${QKFORMER_LUT_CIFAR100_DATA_DIR:-$ROOT/data/cifar100}"

mkdir -p "$DATA_DIR"
"$PYTHON_BIN" - "$DATA_DIR" <<'PY'
import sys
from pathlib import Path
from torchvision import datasets

root = Path(sys.argv[1]).resolve()
train = datasets.CIFAR100(root=str(root), train=True, download=True)
test = datasets.CIFAR100(root=str(root), train=False, download=True)
print(f"[cifar100] root={root}")
print(f"[cifar100] train={len(train)} test={len(test)}")
print(f"[cifar100] extracted={(root / 'cifar-100-python').is_dir()}")
PY
