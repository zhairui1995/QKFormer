#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export QKFORMER_LUT_CKPT="${QKFORMER_LUT_CKPT:-auto}"
bash scripts/server/run_qkformer_lut_e0_diag.sh
