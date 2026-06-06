#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-qk_lutformer_t1_e3_alpha_sweep_artifacts.tar.gz}"
export QKFORMER_LUT_PACKAGE_E3_COUNT="${QKFORMER_LUT_PACKAGE_E3_COUNT:-27}"

bash scripts/server/package_qkformer_lut_t1_e3_seed_sweep.sh "$OUT"
