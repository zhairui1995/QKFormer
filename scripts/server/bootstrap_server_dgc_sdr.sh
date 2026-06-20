#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${QKFORMER_SDR_ENV_PREFIX:-$HOME/.conda_envs/sdr}"
SOURCE_ENV="${QKFORMER_SDR_SOURCE_ENV:-$HOME/.conda_envs/ln}"
CONDA_BIN="${QKFORMER_CONDA_BIN:-/home/mengying/anaconda3/bin/conda}"
PKG_CACHE="${QKFORMER_CONDA_PKGS_DIRS:-$HOME/.conda_pkgs}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "[sdr-env] conda executable not found: $CONDA_BIN" >&2
  exit 1
fi
mkdir -p "$PKG_CACHE"
export CONDA_PKGS_DIRS="$PKG_CACHE"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_PREFIX" --clone "$SOURCE_ENV"
fi

PYTHON_BIN="$ENV_PREFIX/bin/python"
"$PYTHON_BIN" -m pip install --upgrade \
  "pip<25" \
  "setuptools<71" \
  wheel
"$PYTHON_BIN" -m pip install \
  "timm==0.6.12" \
  "PyYAML>=6,<7" \
  "tensorboard>=2.12,<3" \
  "cupy-cuda11x==11.4.0"

"$PYTHON_BIN" - <<'PY'
import torch
import torchvision
import timm
import yaml
import cupy
from spikingjelly.clock_driven import functional

print(f"[sdr-env] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[sdr-env] torchvision={torchvision.__version__} timm={timm.__version__}")
print(f"[sdr-env] pyyaml={yaml.__version__} cuda_available={torch.cuda.is_available()}")
print(f"[sdr-env] cupy={cupy.__version__}")
print(f"[sdr-env] gpu_count={torch.cuda.device_count()}")
PY
