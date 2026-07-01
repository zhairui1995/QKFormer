#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU="${GPU:-${STRUCTURED_CNL_LUT_LIF_GPU:-}}"
if [[ -z "$GPU" ]]; then
  echo "[structured-cnl-lut-lif] GPU is required" >&2
  exit 2
fi

PYTHON_BIN="${STRUCTURED_CNL_LUT_LIF_PYTHON:-/home/lbz/miniconda/envs/sdr/bin/python}"
DATA_DIR="${STRUCTURED_CNL_LUT_LIF_DATA_DIR:-/home/datasets}"
CHECKPOINT="${STRUCTURED_CNL_LUT_LIF_CHECKPOINT:-/home/lbz/pretrained_models/qk_lutformer_retained_20260619/qkformer_cifar100_t4_seed42_model_best.pth.tar}"
FAMILY="${STRUCTURED_CNL_LUT_LIF_FAMILY:-cifar100}"
TIME_STEP="${STRUCTURED_CNL_LUT_LIF_TIME_STEP:-4}"
RUN_MODE="${STRUCTURED_CNL_LUT_LIF_RUN_MODE:-smoke}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${STRUCTURED_CNL_LUT_LIF_RESULT_DIR:-$ROOT/results/structured_residual_cnl_lut_lif_${FAMILY}_t${TIME_STEP}_seed42_${RUN_MODE}_${STAMP}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[structured-cnl-lut-lif] python not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[structured-cnl-lut-lif] checkpoint not found: $CHECKPOINT" >&2
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
  --state-bits "${STRUCTURED_CNL_LUT_LIF_STATE_BITS:-6}"
  --input-bits "${STRUCTURED_CNL_LUT_LIF_INPUT_BITS:-6}"
  --seed "${STRUCTURED_CNL_LUT_LIF_SEED:-42}"
  --max-drop "${STRUCTURED_CNL_LUT_LIF_MAX_DROP:-1.25}"
)

if [[ "${STRUCTURED_CNL_LUT_LIF_LEARN_NORM_AFFINE:-0}" == "1" ]]; then
  COMMON+=(--learn-norm-affine)
else
  COMMON+=(--no-learn-norm-affine)
fi

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
    --calib-batches "${STRUCTURED_CNL_LUT_LIF_CALIB_BATCHES:-32}"
    --epochs "${STRUCTURED_CNL_LUT_LIF_EPOCHS:-3}"
    --max-train-batches "${STRUCTURED_CNL_LUT_LIF_MAX_TRAIN_BATCHES:-128}"
    --amp
  )
else
  echo "[structured-cnl-lut-lif] STRUCTURED_CNL_LUT_LIF_RUN_MODE must be smoke or full, got: $RUN_MODE" >&2
  exit 2
fi

{
  echo "[structured-cnl-lut-lif] host=$(hostname)"
  echo "[structured-cnl-lut-lif] start=$(date -Is)"
  echo "[structured-cnl-lut-lif] root=$ROOT"
  echo "[structured-cnl-lut-lif] result_dir=$RESULT_DIR"
  echo "[structured-cnl-lut-lif] gpu=$GPU"
  echo "[structured-cnl-lut-lif] mode=$RUN_MODE"
  echo "[structured-cnl-lut-lif] checkpoint=$CHECKPOINT"
  "$PYTHON_BIN" tools/structured_residual_cnl_lut_lif_probe.py "${COMMON[@]}" "${EXTRA[@]}"
  echo "[structured-cnl-lut-lif] done=$(date -Is)"
} 2>&1 | tee "$RESULT_DIR/launcher.log"

echo "[structured-cnl-lut-lif] completed result_dir=$RESULT_DIR"
