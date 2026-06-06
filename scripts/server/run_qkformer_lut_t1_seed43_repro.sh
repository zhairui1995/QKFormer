#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export QKFORMER_LUT_TIME_STEP=1
export QKFORMER_TRAIN_SEED="${QKFORMER_TRAIN_SEED:-43}"
export QKFORMER_TRAIN_EXPERIMENT="${QKFORMER_TRAIN_EXPERIMENT:-qkformer_cifar10_t1_seed${QKFORMER_TRAIN_SEED}}"
export QKFORMER_LUT_E3_SEED_SWEEP="${QKFORMER_LUT_E3_SEED_SWEEP:-42,43,44}"
export QKFORMER_LUT_E3_MODE_SWEEP="${QKFORMER_LUT_E3_MODE_SWEEP:-address_lut,global_mean,token_channel_lut}"

echo "[qk-lut-t1-seed-repro] train_seed=$QKFORMER_TRAIN_SEED"
echo "[qk-lut-t1-seed-repro] experiment=$QKFORMER_TRAIN_EXPERIMENT"

bash scripts/server/run_qkformer_cifar10_train.sh "$@"
bash scripts/server/run_qkformer_lut_t1_e0_after_latest_train.sh "$@"
bash scripts/server/run_qkformer_lut_t1_e3_seed_sweep.sh "$@"
