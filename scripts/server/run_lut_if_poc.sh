#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU="${GPU:-${LUT_IF_GPU:-}}"
if [[ -z "$GPU" ]]; then
  echo "[lut-if] GPU is required; pass --gpu through the detached launcher" >&2
  exit 2
fi

PYTHON_BIN="${LUT_IF_PYTHON:-/home/lbz/miniconda/envs/sdr/bin/python}"
DATA_DIR="${LUT_IF_DATA_DIR:-/home/datasets}"
CHECKPOINT="${LUT_IF_CHECKPOINT:-/home/lbz/pretrained_models/qk_lutformer_retained_20260619/qkformer_cifar100_t4_seed42_model_best.pth.tar}"
RUN_MODE="${LUT_IF_RUN_MODE:-full}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${LUT_IF_RESULT_DIR:-$ROOT/results/lut_if_poc_c100_t4_seed42_${RUN_MODE}_${STAMP}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[lut-if] python not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[lut-if] checkpoint not found: $CHECKPOINT" >&2
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
  --max-drop 1.0
)

if [[ "$RUN_MODE" == "smoke" ]]; then
  EXTRA=(
    --batch-size 4
    --workers 2
    --calib-batches 1
    --epochs 1
    --max-train-batches 1
    --max-eval-batches 1
    --no-amp
  )
elif [[ "$RUN_MODE" == "full" ]]; then
  EXTRA=(
    --batch-size 32
    --workers 4
    --calib-batches 32
    --epochs 3
    --max-train-batches 128
    --amp
  )
else
  echo "[lut-if] LUT_IF_RUN_MODE must be smoke or full, got: $RUN_MODE" >&2
  exit 2
fi

{
  echo "[lut-if] root=$ROOT"
  echo "[lut-if] result_dir=$RESULT_DIR"
  echo "[lut-if] gpu=$GPU"
  echo "[lut-if] mode=$RUN_MODE"
  echo "[lut-if] checkpoint=$CHECKPOINT"
  "$PYTHON_BIN" tools/lut_if_poc.py "${COMMON[@]}" "${EXTRA[@]}"
} 2>&1 | tee "$RESULT_DIR/launcher.log"

echo "[lut-if] completed result_dir=$RESULT_DIR"
