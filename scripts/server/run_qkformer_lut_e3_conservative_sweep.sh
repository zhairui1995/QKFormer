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

export QKFORMER_LUT_E3_EPOCHS="${QKFORMER_LUT_E3_EPOCHS:-2}"
export QKFORMER_LUT_E3_CALIB_BATCHES="${QKFORMER_LUT_E3_CALIB_BATCHES:-128}"
export QKFORMER_LUT_E3_TRAIN_BATCHES="${QKFORMER_LUT_E3_TRAIN_BATCHES:-128}"
export QKFORMER_LUT_E3_EVAL_BATCHES="${QKFORMER_LUT_E3_EVAL_BATCHES:-0}"
export QKFORMER_LUT_E3_LR="${QKFORMER_LUT_E3_LR:-0.003}"
export QKFORMER_LUT_E3_ALPHA_INIT="${QKFORMER_LUT_E3_ALPHA_INIT:-0.1}"
export QKFORMER_LUT_E3_LEARN_ALPHA="${QKFORMER_LUT_E3_LEARN_ALPHA:-0}"
export QKFORMER_LUT_E3_LAMBDA_KL="${QKFORMER_LUT_E3_LAMBDA_KL:-2.0}"
export QKFORMER_LUT_E3_LAMBDA_LOCAL_MSE="${QKFORMER_LUT_E3_LAMBDA_LOCAL_MSE:-0.2}"

echo "[qk-lut-e3-conservative-sweep] root=$ROOT"
echo "[qk-lut-e3-conservative-sweep] targets=$TARGETS"
echo "[qk-lut-e3-conservative-sweep] mode_sweep=$MODE_SWEEP"
echo "[qk-lut-e3-conservative-sweep] seed_sweep=$SEED_SWEEP"
echo "[qk-lut-e3-conservative-sweep] epochs=$QKFORMER_LUT_E3_EPOCHS"
echo "[qk-lut-e3-conservative-sweep] lr=$QKFORMER_LUT_E3_LR"
echo "[qk-lut-e3-conservative-sweep] alpha_init=$QKFORMER_LUT_E3_ALPHA_INIT"
echo "[qk-lut-e3-conservative-sweep] learn_alpha=$QKFORMER_LUT_E3_LEARN_ALPHA"
echo "[qk-lut-e3-conservative-sweep] lambda_kl=$QKFORMER_LUT_E3_LAMBDA_KL"
echo "[qk-lut-e3-conservative-sweep] lambda_local_mse=$QKFORMER_LUT_E3_LAMBDA_LOCAL_MSE"
echo "[qk-lut-e3-conservative-sweep] calib_batches=$QKFORMER_LUT_E3_CALIB_BATCHES"
echo "[qk-lut-e3-conservative-sweep] train_batches=$QKFORMER_LUT_E3_TRAIN_BATCHES"
echo "[qk-lut-e3-conservative-sweep] eval_batches=$QKFORMER_LUT_E3_EVAL_BATCHES"

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
    echo "[qk-lut-e3-conservative-sweep] run_mode=$mode seed=$seed targets=$TARGETS"
    QKFORMER_LUT_E3_MODE="$mode" \
    QKFORMER_LUT_E3_SEED="$seed" \
    QKFORMER_LUT_E3_TARGETS="$TARGETS" \
      bash scripts/server/run_qkformer_lut_e3_trainable_lut.sh
  done
done

echo "[qk-lut-e3-conservative-sweep] done=$(date -Is)"
