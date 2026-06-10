#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
export QKFORMER_LUT_GPU="${QKFORMER_LUT_GPU:-2}"
qk_lut_configure_gpu

SCALE_SWEEP="${QKFORMER_LUT_E3_ADDRESS_SCALE_SWEEP:-0.1,0.25,0.5}"
export QKFORMER_LUT_E3_CONFIG=configs/qkformer_lut_cifar100_t1_e3_trainable_lut.yaml
export QKFORMER_LUT_TIME_STEP=1
export QKFORMER_LUT_DATA_DIR="${QKFORMER_LUT_CIFAR100_DATA_DIR:-$ROOT/data/cifar100}"
export QKFORMER_LUT_E3_ALPHA_INIT=0.025
export QKFORMER_LUT_E3_SWEEP_TARGETS=stage1.0.tssa
export QKFORMER_LUT_E3_MODE_SWEEP=global_plus_address_lut,global_plus_shuffled_address_lut
export QKFORMER_LUT_E3_SEED_SWEEP=42,43,44

echo "[qk-lut-c100-address-scale] scales=$SCALE_SWEEP"
IFS=',' read -r -a SCALE_GROUPS <<< "$SCALE_SWEEP"
for scale in "${SCALE_GROUPS[@]}"; do
  scale="$(echo "$scale" | xargs)"
  [[ -n "$scale" ]] || continue
  echo "[qk-lut-c100-address-scale] run_scale=$scale"
  QKFORMER_LUT_E3_ADDRESS_SCALE="$scale" \
    bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh --gpu "$QKFORMER_LUT_GPU"
done

QKFORMER_LUT_PACKAGE_E3_COUNT=18 \
  bash scripts/server/package_qkformer_lut_cifar100_t1.sh \
    qk_lutformer_cifar100_t1_address_scale_sweep_artifacts.tar.gz
echo "[qk-lut-c100-address-scale] done=$(date -Is)"
