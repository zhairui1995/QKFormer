#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export QKFORMER_LUT_TIME_STEP=1
export QKFORMER_TRAIN_SEED="${QKFORMER_TRAIN_SEED:-44}"
export QKFORMER_TRAIN_EXPERIMENT="${QKFORMER_TRAIN_EXPERIMENT:-qkformer_cifar10_t1_seed${QKFORMER_TRAIN_SEED}}"

export QKFORMER_LUT_E3_ALPHA_SWEEP="${QKFORMER_LUT_E3_ALPHA_SWEEP:-0.025}"
export QKFORMER_LUT_E3_MODE_SWEEP="${QKFORMER_LUT_E3_MODE_SWEEP:-address_lut,global_mean,token_channel_lut,shuffled_address_lut}"
export QKFORMER_LUT_E3_SEED_SWEEP="${QKFORMER_LUT_E3_SEED_SWEEP:-42,43,44}"

ARTIFACT="${QKFORMER_LUT_REPRO_ARTIFACT:-qk_lutformer_t1_seed${QKFORMER_TRAIN_SEED}_alpha0025_repro_artifacts.tar.gz}"

echo "[qk-lut-t1-seed44-alpha0025] train_seed=$QKFORMER_TRAIN_SEED"
echo "[qk-lut-t1-seed44-alpha0025] experiment=$QKFORMER_TRAIN_EXPERIMENT"
echo "[qk-lut-t1-seed44-alpha0025] alpha_sweep=$QKFORMER_LUT_E3_ALPHA_SWEEP"
echo "[qk-lut-t1-seed44-alpha0025] mode_sweep=$QKFORMER_LUT_E3_MODE_SWEEP"
echo "[qk-lut-t1-seed44-alpha0025] e3_seed_sweep=$QKFORMER_LUT_E3_SEED_SWEEP"

bash scripts/server/run_qkformer_cifar10_train.sh "$@"
bash scripts/server/run_qkformer_lut_t1_e0_after_latest_train.sh "$@"
bash scripts/server/run_qkformer_lut_t1_e3_alpha_sweep.sh "$@"

QKFORMER_LUT_PACKAGE_E3_COUNT="${QKFORMER_LUT_PACKAGE_E3_COUNT:-12}" \
  bash scripts/server/package_qkformer_lut_t1_e3_seed_sweep.sh "$ARTIFACT"

echo "[qk-lut-t1-seed44-alpha0025] artifact=$ROOT/$ARTIFACT"
echo "[qk-lut-t1-seed44-alpha0025] done=$(date -Is)"
