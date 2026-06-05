#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu

export QKFORMER_LUT_E2_TARGETS="${QKFORMER_LUT_E2_TARGETS:-stage1.0.tssa}"
export QKFORMER_LUT_E2_CALIB_BATCHES="${QKFORMER_LUT_E2_CALIB_BATCHES:-0}"
export QKFORMER_LUT_E2_EVAL_BATCHES="${QKFORMER_LUT_E2_EVAL_BATCHES:-0}"
BLEND_SWEEP="${QKFORMER_LUT_E2_BLEND_SWEEP:-0,0.25,0.5,0.75,1.0}"

echo "[qk-lut-e2-blend-sweep] root=$ROOT"
echo "[qk-lut-e2-blend-sweep] targets=$QKFORMER_LUT_E2_TARGETS"
echo "[qk-lut-e2-blend-sweep] calib_batches=$QKFORMER_LUT_E2_CALIB_BATCHES"
echo "[qk-lut-e2-blend-sweep] eval_batches=$QKFORMER_LUT_E2_EVAL_BATCHES"
echo "[qk-lut-e2-blend-sweep] blend_sweep=$BLEND_SWEEP"

IFS=',' read -r -a BLEND_GROUPS <<< "$BLEND_SWEEP"
for blend in "${BLEND_GROUPS[@]}"; do
  blend="$(echo "$blend" | xargs)"
  if [[ -z "$blend" ]]; then
    continue
  fi
  echo "[qk-lut-e2-blend-sweep] run_blend=$blend"
  QKFORMER_LUT_E2_BLEND="$blend" bash scripts/server/run_qkformer_lut_e2_replace.sh
done

echo "[qk-lut-e2-blend-sweep] done=$(date -Is)"
