#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu

TARGETS="${QKFORMER_LUT_E2_RANDOM_CALIB_TARGETS:-stage1.0.tssa}"
CALIB_SWEEP="${QKFORMER_LUT_E2_RANDOM_CALIB_BATCH_SWEEP:-128,512}"
BLEND_SWEEP="${QKFORMER_LUT_E2_RANDOM_CALIB_BLEND_SWEEP:-0.25,1.0}"
SEED_SWEEP="${QKFORMER_LUT_E2_RANDOM_CALIB_SEED_SWEEP:-42,43,44}"
export QKFORMER_LUT_E2_EVAL_BATCHES="${QKFORMER_LUT_E2_EVAL_BATCHES:-0}"
export QKFORMER_LUT_E2_CALIB_SHUFFLE="${QKFORMER_LUT_E2_CALIB_SHUFFLE:-1}"

echo "[qk-lut-e2-random-calib-sweep] root=$ROOT"
echo "[qk-lut-e2-random-calib-sweep] targets=$TARGETS"
echo "[qk-lut-e2-random-calib-sweep] calib_sweep=$CALIB_SWEEP"
echo "[qk-lut-e2-random-calib-sweep] blend_sweep=$BLEND_SWEEP"
echo "[qk-lut-e2-random-calib-sweep] seed_sweep=$SEED_SWEEP"
echo "[qk-lut-e2-random-calib-sweep] eval_batches=$QKFORMER_LUT_E2_EVAL_BATCHES"
echo "[qk-lut-e2-random-calib-sweep] calib_shuffle=$QKFORMER_LUT_E2_CALIB_SHUFFLE"

IFS=',' read -r -a CALIB_GROUPS <<< "$CALIB_SWEEP"
IFS=',' read -r -a BLEND_GROUPS <<< "$BLEND_SWEEP"
IFS=',' read -r -a SEED_GROUPS <<< "$SEED_SWEEP"

for calib_batches in "${CALIB_GROUPS[@]}"; do
  calib_batches="$(echo "$calib_batches" | xargs)"
  if [[ -z "$calib_batches" ]]; then
    continue
  fi
  for blend in "${BLEND_GROUPS[@]}"; do
    blend="$(echo "$blend" | xargs)"
    if [[ -z "$blend" ]]; then
      continue
    fi
    for seed in "${SEED_GROUPS[@]}"; do
      seed="$(echo "$seed" | xargs)"
      if [[ -z "$seed" ]]; then
        continue
      fi
      echo "[qk-lut-e2-random-calib-sweep] run_calib_batches=$calib_batches blend=$blend seed=$seed targets=$TARGETS"
      QKFORMER_LUT_E2_TARGETS="$TARGETS" \
      QKFORMER_LUT_E2_CALIB_BATCHES="$calib_batches" \
      QKFORMER_LUT_E2_BLEND="$blend" \
      QKFORMER_LUT_E2_CALIB_SEED="$seed" \
        bash scripts/server/run_qkformer_lut_e2_replace.sh
    done
  done
done

echo "[qk-lut-e2-random-calib-sweep] done=$(date -Is)"
