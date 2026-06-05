#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export QKFORMER_LUT_TIME_STEP=1

bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh "$@"
