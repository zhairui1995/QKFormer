#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu

TARGETS="${QKFORMER_LUT_E3_SWEEP_TARGETS:-stage1.0.tssa}"
MODE_SWEEP="${QKFORMER_LUT_E3_MODE_SWEEP:-address_lut,global_mean,token_channel_lut}"
SEED_SWEEP="${QKFORMER_LUT_E3_SEED_SWEEP:-42}"
export QKFORMER_LUT_E3_EPOCHS="${QKFORMER_LUT_E3_EPOCHS:-5}"
export QKFORMER_LUT_E3_CALIB_BATCHES="${QKFORMER_LUT_E3_CALIB_BATCHES:-128}"
export QKFORMER_LUT_E3_TRAIN_BATCHES="${QKFORMER_LUT_E3_TRAIN_BATCHES:-128}"
export QKFORMER_LUT_E3_EVAL_BATCHES="${QKFORMER_LUT_E3_EVAL_BATCHES:-0}"

echo "[qk-lut-e3-sweep] root=$ROOT"
echo "[qk-lut-e3-sweep] targets=$TARGETS"
echo "[qk-lut-e3-sweep] mode_sweep=$MODE_SWEEP"
echo "[qk-lut-e3-sweep] seed_sweep=$SEED_SWEEP"
echo "[qk-lut-e3-sweep] epochs=$QKFORMER_LUT_E3_EPOCHS"
echo "[qk-lut-e3-sweep] calib_batches=$QKFORMER_LUT_E3_CALIB_BATCHES"
echo "[qk-lut-e3-sweep] train_batches=$QKFORMER_LUT_E3_TRAIN_BATCHES"
echo "[qk-lut-e3-sweep] eval_batches=$QKFORMER_LUT_E3_EVAL_BATCHES"

IFS=',' read -r -a MODE_GROUPS <<< "$MODE_SWEEP"
IFS=',' read -r -a SEED_GROUPS <<< "$SEED_SWEEP"

for mode in "${MODE_GROUPS[@]}"; do
  mode="$(echo "$mode" | xargs)"
  if [[ -z "$mode" ]]; then
    continue
  fi
  for seed in "${SEED_GROUPS[@]}"; do
    seed="$(echo "$seed" | xargs)"
    if [[ -z "$seed" ]]; then
      continue
    fi
    echo "[qk-lut-e3-sweep] run_mode=$mode seed=$seed targets=$TARGETS"
    QKFORMER_LUT_E3_MODE="$mode" \
    QKFORMER_LUT_E3_SEED="$seed" \
    QKFORMER_LUT_E3_TARGETS="$TARGETS" \
      bash scripts/server/run_qkformer_lut_e3_trainable_lut.sh
  done
done

echo "[qk-lut-e3-sweep] done=$(date -Is)"
