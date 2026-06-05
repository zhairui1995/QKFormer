#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu

TARGETS="${QKFORMER_LUT_E2_CALIB_SWEEP_TARGETS:-stage1.0.tssa}"
CALIB_SWEEP="${QKFORMER_LUT_E2_CALIB_SWEEP:-32,128,512,0}"
export QKFORMER_LUT_E2_EVAL_BATCHES="${QKFORMER_LUT_E2_EVAL_BATCHES:-0}"

echo "[qk-lut-e2-calib-sweep] root=$ROOT"
echo "[qk-lut-e2-calib-sweep] targets=$TARGETS"
echo "[qk-lut-e2-calib-sweep] calib_sweep=$CALIB_SWEEP"
echo "[qk-lut-e2-calib-sweep] eval_batches=$QKFORMER_LUT_E2_EVAL_BATCHES"

IFS=',' read -r -a CALIB_GROUPS <<< "$CALIB_SWEEP"
for calib_batches in "${CALIB_GROUPS[@]}"; do
  calib_batches="$(echo "$calib_batches" | xargs)"
  if [[ -z "$calib_batches" ]]; then
    continue
  fi
  echo "[qk-lut-e2-calib-sweep] run_calib_batches=$calib_batches targets=$TARGETS"
  QKFORMER_LUT_E2_TARGETS="$TARGETS" \
  QKFORMER_LUT_E2_CALIB_BATCHES="$calib_batches" \
    bash scripts/server/run_qkformer_lut_e2_replace.sh
done

echo "[qk-lut-e2-calib-sweep] done=$(date -Is)"
