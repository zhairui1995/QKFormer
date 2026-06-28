#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU="${GPU:-${CNL_LUT_LIF_GPU:-}}"
if [[ -z "$GPU" ]]; then
  echo "[cnl-lut-lif] GPU is required" >&2
  exit 2
fi

PYTHON_BIN="${CNL_LUT_LIF_PYTHON:-/home/lbz/miniconda/envs/sdr/bin/python}"
DATA_DIR="${CNL_LUT_LIF_DATA_DIR:-/home/datasets}"
CHECKPOINT="${CNL_LUT_LIF_CHECKPOINT:-/home/lbz/pretrained_models/qk_lutformer_retained_20260619/qkformer_cifar100_t4_seed42_model_best.pth.tar}"
RUN_MODE="${CNL_LUT_LIF_RUN_MODE:-full}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${CNL_LUT_LIF_RESULT_DIR:-$ROOT/results/cnl_lut_lif_e0_c100_t4_seed42_${RUN_MODE}_${STAMP}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[cnl-lut-lif] python not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[cnl-lut-lif] checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

COMMON=(
  --root "$ROOT"
  --data-dir "$DATA_DIR"
  --checkpoint "$CHECKPOINT"
  --result-dir "$RESULT_DIR"
  --local-cuda-index 0
  --time-step 4
  --state-bits 6
  --input-bits 6
  --seed 42
)

if [[ "$RUN_MODE" == "smoke" ]]; then
  EXTRA=(
    --batch-size 4
    --workers 2
    --calib-batches 1
    --max-eval-batches 1
  )
elif [[ "$RUN_MODE" == "full" ]]; then
  EXTRA=(
    --batch-size 32
    --workers 4
    --calib-batches 128
  )
else
  echo "[cnl-lut-lif] CNL_LUT_LIF_RUN_MODE must be smoke or full, got: $RUN_MODE" >&2
  exit 2
fi

{
  echo "[cnl-lut-lif] root=$ROOT"
  echo "[cnl-lut-lif] result_dir=$RESULT_DIR"
  echo "[cnl-lut-lif] gpu=$GPU"
  echo "[cnl-lut-lif] mode=$RUN_MODE"
  echo "[cnl-lut-lif] checkpoint=$CHECKPOINT"
  "$PYTHON_BIN" tools/cnl_lut_lif_e0.py "${COMMON[@]}" "${EXTRA[@]}"
} 2>&1 | tee "$RESULT_DIR/launcher.log"

echo "[cnl-lut-lif] completed result_dir=$RESULT_DIR"
