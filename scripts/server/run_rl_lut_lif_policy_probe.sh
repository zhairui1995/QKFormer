#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU="${GPU:-${RL_LUT_LIF_GPU:-}}"
if [[ -z "$GPU" ]]; then
  echo "[rl-lut-lif] GPU is required" >&2
  exit 2
fi

PYTHON_BIN="${RL_LUT_LIF_PYTHON:-/home/lbz/miniconda/envs/sdr/bin/python}"
DATA_DIR="${RL_LUT_LIF_DATA_DIR:-/home/datasets}"
CHECKPOINT="${RL_LUT_LIF_CHECKPOINT:-/home/lbz/pretrained_models/qk_lutformer_retained_20260619/qkformer_cifar100_t4_seed42_model_best.pth.tar}"
FAMILY="${RL_LUT_LIF_FAMILY:-cifar100}"
TIME_STEP="${RL_LUT_LIF_TIME_STEP:-4}"
RUN_MODE="${RL_LUT_LIF_RUN_MODE:-smoke}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RL_LUT_LIF_RESULT_DIR:-$ROOT/results/rl_qklut_lif_policy_${FAMILY}_t${TIME_STEP}_seed42_${RUN_MODE}_${STAMP}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[rl-lut-lif] python not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[rl-lut-lif] checkpoint not found: $CHECKPOINT" >&2
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
  --state-bits "${RL_LUT_LIF_STATE_BITS:-6}"
  --input-bits "${RL_LUT_LIF_INPUT_BITS:-6}"
  --seed "${RL_LUT_LIF_SEED:-42}"
  --policy-rounds "${RL_LUT_LIF_POLICY_ROUNDS:-1}"
  --max-drop "${RL_LUT_LIF_MAX_DROP:-1.50}"
)

if [[ "$RUN_MODE" == "smoke" ]]; then
  EXTRA=(
    --batch-size 4
    --workers 2
    --calib-batches 1
    --search-eval-batches 1
    --final-eval-batches 1
  )
elif [[ "$RUN_MODE" == "full" ]]; then
  EXTRA=(
    --batch-size 32
    --workers 4
    --calib-batches "${RL_LUT_LIF_CALIB_BATCHES:-128}"
    --search-eval-batches "${RL_LUT_LIF_SEARCH_EVAL_BATCHES:-8}"
    --final-eval-batches -1
  )
else
  echo "[rl-lut-lif] RL_LUT_LIF_RUN_MODE must be smoke or full, got: $RUN_MODE" >&2
  exit 2
fi

{
  echo "[rl-lut-lif] host=$(hostname)"
  echo "[rl-lut-lif] start=$(date -Is)"
  echo "[rl-lut-lif] root=$ROOT"
  echo "[rl-lut-lif] result_dir=$RESULT_DIR"
  echo "[rl-lut-lif] gpu=$GPU"
  echo "[rl-lut-lif] mode=$RUN_MODE"
  echo "[rl-lut-lif] checkpoint=$CHECKPOINT"
  "$PYTHON_BIN" tools/rl_lut_lif_policy_probe.py "${COMMON[@]}" "${EXTRA[@]}"
  echo "[rl-lut-lif] done=$(date -Is)"
} 2>&1 | tee "$RESULT_DIR/launcher.log"

echo "[rl-lut-lif] completed result_dir=$RESULT_DIR"
