#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU="${GPU:-${TRAINABLE_CNL_LUT_LIF_GPU:-}}"
if [[ -z "$GPU" ]]; then
  echo "[trainable-cnl-lut-lif] GPU is required" >&2
  exit 2
fi

PYTHON_BIN="${TRAINABLE_CNL_LUT_LIF_PYTHON:-/home/lbz/miniconda/envs/sdr/bin/python}"
DATA_DIR="${TRAINABLE_CNL_LUT_LIF_DATA_DIR:-/home/datasets}"
CHECKPOINT="${TRAINABLE_CNL_LUT_LIF_CHECKPOINT:-/home/lbz/pretrained_models/qk_lutformer_retained_20260619/qkformer_cifar100_t4_seed42_model_best.pth.tar}"
FAMILY="${TRAINABLE_CNL_LUT_LIF_FAMILY:-cifar100}"
TIME_STEP="${TRAINABLE_CNL_LUT_LIF_TIME_STEP:-4}"
RUN_MODE="${TRAINABLE_CNL_LUT_LIF_RUN_MODE:-smoke}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${TRAINABLE_CNL_LUT_LIF_RESULT_DIR:-$ROOT/results/trainable_cnl_lut_lif_${FAMILY}_t${TIME_STEP}_seed42_${RUN_MODE}_${STAMP}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[trainable-cnl-lut-lif] python not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[trainable-cnl-lut-lif] checkpoint not found: $CHECKPOINT" >&2
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
  --family "$FAMILY"
  --local-cuda-index 0
  --time-step "$TIME_STEP"
  --state-bits "${TRAINABLE_CNL_LUT_LIF_STATE_BITS:-6}"
  --input-bits "${TRAINABLE_CNL_LUT_LIF_INPUT_BITS:-6}"
  --seed "${TRAINABLE_CNL_LUT_LIF_SEED:-42}"
  --max-drop "${TRAINABLE_CNL_LUT_LIF_MAX_DROP:-1.25}"
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
    --calib-batches "${TRAINABLE_CNL_LUT_LIF_CALIB_BATCHES:-32}"
    --epochs "${TRAINABLE_CNL_LUT_LIF_EPOCHS:-3}"
    --max-train-batches "${TRAINABLE_CNL_LUT_LIF_MAX_TRAIN_BATCHES:-128}"
    --amp
  )
else
  echo "[trainable-cnl-lut-lif] TRAINABLE_CNL_LUT_LIF_RUN_MODE must be smoke or full, got: $RUN_MODE" >&2
  exit 2
fi

{
  echo "[trainable-cnl-lut-lif] host=$(hostname)"
  echo "[trainable-cnl-lut-lif] start=$(date -Is)"
  echo "[trainable-cnl-lut-lif] root=$ROOT"
  echo "[trainable-cnl-lut-lif] result_dir=$RESULT_DIR"
  echo "[trainable-cnl-lut-lif] gpu=$GPU"
  echo "[trainable-cnl-lut-lif] mode=$RUN_MODE"
  echo "[trainable-cnl-lut-lif] checkpoint=$CHECKPOINT"
  "$PYTHON_BIN" tools/trainable_cnl_lut_lif_probe.py "${COMMON[@]}" "${EXTRA[@]}"
  echo "[trainable-cnl-lut-lif] done=$(date -Is)"
} 2>&1 | tee "$RESULT_DIR/launcher.log"

echo "[trainable-cnl-lut-lif] completed result_dir=$RESULT_DIR"
