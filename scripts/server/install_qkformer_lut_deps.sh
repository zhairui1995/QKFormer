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
    echo "[qk-lut-deps] missing python3/python"
    exit 1
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$ROOT/results/qkformer_lut_deps_${TS}"
LOG_FILE="$RESULT_DIR/install_log.txt"
mkdir -p "$RESULT_DIR"

{
  echo "[qk-lut-deps] root=$ROOT"
  echo "[qk-lut-deps] result_dir=$RESULT_DIR"
  echo "[qk-lut-deps] start=$(date -Is)"
  echo "[qk-lut-deps] python=$PYTHON_BIN"
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" -m pip --version

  CUDA_VERSION="$("$PYTHON_BIN" - <<'PY'
try:
    import torch
    print(torch.version.cuda or "")
except Exception:
    print("")
PY
)"
  if [[ -z "$CUDA_VERSION" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  fi
  echo "[qk-lut-deps] detected_cuda=$CUDA_VERSION"

  CUPY_PACKAGE="${CUPY_PACKAGE:-}"
  if [[ -z "$CUPY_PACKAGE" ]]; then
    case "$CUDA_VERSION" in
      10.*)
        CUPY_PACKAGE="cupy-cuda102==11.4.0"
        ;;
      11.*|"")
        CUPY_PACKAGE="cupy-cuda11x==11.4.0"
        ;;
      12.*)
        CUPY_PACKAGE="cupy-cuda12x"
        ;;
      *)
        CUPY_PACKAGE="cupy-cuda11x==11.4.0"
        ;;
    esac
  fi
  echo "[qk-lut-deps] cupy_package=$CUPY_PACKAGE"

  if [[ "${SKIP_PIP_UPGRADE:-0}" != "1" ]]; then
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  fi

  "$PYTHON_BIN" -m pip install \
    "spikingjelly==0.0.0.0.12" \
    "$CUPY_PACKAGE"

  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = ["torch", "yaml", "timm", "torchvision", "spikingjelly", "cupy"]
missing = []
for name in required:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"[qk-lut-deps] import_ok {name} {version}")
    except Exception as exc:
        missing.append((name, str(exc)))

if missing:
    for name, exc in missing:
        print(f"[qk-lut-deps] import_failed {name}: {exc}")
    sys.exit(1)

import torch
if not torch.cuda.is_available():
    print("[qk-lut-deps] warning: torch.cuda.is_available() is false")
else:
    print(f"[qk-lut-deps] torch_cuda={torch.version.cuda}")

import cupy
try:
    device_count = cupy.cuda.runtime.getDeviceCount()
    print(f"[qk-lut-deps] cupy_cuda_devices={device_count}")
except Exception as exc:
    print(f"[qk-lut-deps] warning: cupy CUDA runtime check failed: {exc}")
PY

  echo "[qk-lut-deps] done=$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
