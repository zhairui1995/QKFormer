#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
export QKFORMER_LUT_GPU="${QKFORMER_LUT_GPU:-2}"
qk_lut_configure_gpu

ALPHA_SWEEP="${QKFORMER_LUT_E3_ALPHA_SWEEP:-0.025,0.05,0.1}"
MODE_SWEEP="${QKFORMER_LUT_E3_MODE_SWEEP:-address_lut,global_mean,token_channel_lut}"
SEED_SWEEP="${QKFORMER_LUT_E3_SEED_SWEEP:-42,43,44}"

export QKFORMER_LUT_TIME_STEP=1
export QKFORMER_LUT_E3_MODE_SWEEP="$MODE_SWEEP"
export QKFORMER_LUT_E3_SEED_SWEEP="$SEED_SWEEP"
export QKFORMER_LUT_E3_EPOCHS="${QKFORMER_LUT_E3_EPOCHS:-2}"
export QKFORMER_LUT_E3_CALIB_BATCHES="${QKFORMER_LUT_E3_CALIB_BATCHES:-128}"
export QKFORMER_LUT_E3_TRAIN_BATCHES="${QKFORMER_LUT_E3_TRAIN_BATCHES:-128}"
export QKFORMER_LUT_E3_EVAL_BATCHES="${QKFORMER_LUT_E3_EVAL_BATCHES:-0}"
export QKFORMER_LUT_E3_LR="${QKFORMER_LUT_E3_LR:-0.003}"
export QKFORMER_LUT_E3_LEARN_ALPHA="${QKFORMER_LUT_E3_LEARN_ALPHA:-0}"
export QKFORMER_LUT_E3_LAMBDA_KL="${QKFORMER_LUT_E3_LAMBDA_KL:-2.0}"
export QKFORMER_LUT_E3_LAMBDA_LOCAL_MSE="${QKFORMER_LUT_E3_LAMBDA_LOCAL_MSE:-0.2}"

echo "[qk-lut-t1-e3-alpha-sweep] root=$ROOT"
echo "[qk-lut-t1-e3-alpha-sweep] alpha_sweep=$ALPHA_SWEEP"
echo "[qk-lut-t1-e3-alpha-sweep] mode_sweep=$MODE_SWEEP"
echo "[qk-lut-t1-e3-alpha-sweep] seed_sweep=$SEED_SWEEP"
echo "[qk-lut-t1-e3-alpha-sweep] epochs=$QKFORMER_LUT_E3_EPOCHS"
echo "[qk-lut-t1-e3-alpha-sweep] lr=$QKFORMER_LUT_E3_LR"
echo "[qk-lut-t1-e3-alpha-sweep] learn_alpha=$QKFORMER_LUT_E3_LEARN_ALPHA"

IFS=',' read -r -a ALPHA_GROUPS <<< "$ALPHA_SWEEP"
for alpha in "${ALPHA_GROUPS[@]}"; do
  alpha="$(echo "$alpha" | xargs)"
  if [[ -z "$alpha" ]]; then
    continue
  fi
  echo "[qk-lut-t1-e3-alpha-sweep] run_alpha=$alpha"
  QKFORMER_LUT_E3_ALPHA_INIT="$alpha" \
    bash scripts/server/run_qkformer_lut_t1_e3_seed_sweep.sh --gpu "$QKFORMER_LUT_GPU"
done

echo "[qk-lut-t1-e3-alpha-sweep] done=$(date -Is)"
