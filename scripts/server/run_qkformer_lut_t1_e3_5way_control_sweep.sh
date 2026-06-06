#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export QKFORMER_LUT_TIME_STEP=1
export QKFORMER_LUT_E3_SEED_SWEEP="${QKFORMER_LUT_E3_SEED_SWEEP:-42,43,44}"
export QKFORMER_LUT_E3_MODE_SWEEP="${QKFORMER_LUT_E3_MODE_SWEEP:-address_lut,global_mean,token_channel_lut,shuffled_address_lut}"

echo "[qk-lut-t1-e3-5way] baseline is measured inside each E3 row"
echo "[qk-lut-t1-e3-5way] adapter_modes=$QKFORMER_LUT_E3_MODE_SWEEP"
echo "[qk-lut-t1-e3-5way] seeds=$QKFORMER_LUT_E3_SEED_SWEEP"

bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh "$@"
