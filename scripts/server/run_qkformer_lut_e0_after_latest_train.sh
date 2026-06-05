#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu

export QKFORMER_LUT_CKPT="${QKFORMER_LUT_CKPT:-auto}"
bash scripts/server/run_qkformer_lut_e0_diag.sh
