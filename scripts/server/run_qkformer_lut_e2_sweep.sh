#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu

export QKFORMER_LUT_E2_CALIB_BATCHES="${QKFORMER_LUT_E2_CALIB_BATCHES:-32}"
export QKFORMER_LUT_E2_EVAL_BATCHES="${QKFORMER_LUT_E2_EVAL_BATCHES:-0}"

SWEEP_TARGETS="${QKFORMER_LUT_E2_SWEEP_TARGETS:-stage1.0.tssa;stage2.0.tssa;stage1.0.tssa,stage2.0.tssa}"

echo "[qk-lut-e2-sweep] root=$ROOT"
echo "[qk-lut-e2-sweep] calib_batches=$QKFORMER_LUT_E2_CALIB_BATCHES"
echo "[qk-lut-e2-sweep] eval_batches=$QKFORMER_LUT_E2_EVAL_BATCHES"
echo "[qk-lut-e2-sweep] targets=$SWEEP_TARGETS"

IFS=';' read -r -a TARGET_GROUPS <<< "$SWEEP_TARGETS"
for targets in "${TARGET_GROUPS[@]}"; do
  if [[ -z "$targets" ]]; then
    continue
  fi
  echo "[qk-lut-e2-sweep] run_targets=$targets"
  QKFORMER_LUT_E2_TARGETS="$targets" bash scripts/server/run_qkformer_lut_e2_replace.sh
done

echo "[qk-lut-e2-sweep] done=$(date -Is)"
