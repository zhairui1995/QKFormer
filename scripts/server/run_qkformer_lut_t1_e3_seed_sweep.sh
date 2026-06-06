#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export QKFORMER_LUT_TIME_STEP=1
export QKFORMER_LUT_E3_SEED_SWEEP="${QKFORMER_LUT_E3_SEED_SWEEP:-42,43,44}"
export QKFORMER_LUT_E3_MODE_SWEEP="${QKFORMER_LUT_E3_MODE_SWEEP:-address_lut,global_mean,token_channel_lut}"

bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh "$@"
