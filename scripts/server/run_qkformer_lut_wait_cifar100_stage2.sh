#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MIN_FREE_MB="${QKFORMER_LUT_WAIT_MIN_FREE_MB:-12000}"
MAX_UTIL="${QKFORMER_LUT_WAIT_MAX_UTIL:-50}"
POLL_SECONDS="${QKFORMER_LUT_WAIT_POLL_SECONDS:-60}"
CHECKPOINT="${QKFORMER_LUT_STAGE2_CKPT:-$ROOT/results/qkformer_cifar10_train_20260606_143944/output/qkformer_cifar10_t1_seed43/model_best.pth.tar}"

mkdir -p "$ROOT/results"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
C100_LOG="$ROOT/results/qk_cifar100_t1_train_wait_${RUN_TS}.log"
STAGE2_LOG="$ROOT/results/qk_lut_t1_stage2_alpha0025_wait_${RUN_TS}.log"

echo "[qk-lut-wait-parallel] root=$ROOT"
echo "[qk-lut-wait-parallel] min_free_mb=$MIN_FREE_MB max_util=$MAX_UTIL poll_seconds=$POLL_SECONDS"
echo "[qk-lut-wait-parallel] stage2_checkpoint=$CHECKPOINT"
echo "[qk-lut-wait-parallel] c100_log=$C100_LOG"
echo "[qk-lut-wait-parallel] stage2_log=$STAGE2_LOG"

while true; do
  mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
      awk -F, -v min_free="$MIN_FREE_MB" -v max_util="$MAX_UTIL" '
        {
          gsub(/ /, "", $1);
          free = $2 + 0;
          util = $3 + 0;
          if ($1 != "0" && free >= min_free && util <= max_util) {
            print $1;
          }
        }' |
      head -2
  )
  if [[ "${#GPUS[@]}" -ge 2 ]]; then
    break
  fi
  echo "[qk-lut-wait-parallel] waiting=$(date -Is) available=${GPUS[*]:-none}"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
  sleep "$POLL_SECONDS"
done

C100_GPU="${GPUS[0]}"
STAGE2_GPU="${GPUS[1]}"
echo "[qk-lut-wait-parallel] selected_cifar100_gpu=$C100_GPU"
echo "[qk-lut-wait-parallel] selected_stage2_gpu=$STAGE2_GPU"

(
  cd "$ROOT"
  QKFORMER_LUT_TIME_STEP=1 \
  QKFORMER_CIFAR100_TRAIN_SEED="${QKFORMER_CIFAR100_TRAIN_SEED:-42}" \
    bash scripts/server/run_qkformer_cifar100_train.sh --gpu "$C100_GPU"
) > "$C100_LOG" 2>&1 &
C100_PID="$!"

(
  cd "$ROOT"
  QKFORMER_LUT_CKPT="$CHECKPOINT" \
  QKFORMER_LUT_TIME_STEP=1 \
  QKFORMER_LUT_E3_ALPHA_INIT=0.025 \
  QKFORMER_LUT_E3_SWEEP_TARGETS=stage2.0.tssa \
  QKFORMER_LUT_E3_MODE_SWEEP=address_lut,global_mean,token_channel_lut,shuffled_address_lut \
  QKFORMER_LUT_E3_SEED_SWEEP=42,43,44 \
    bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh --gpu "$STAGE2_GPU"
) > "$STAGE2_LOG" 2>&1 &
STAGE2_PID="$!"

echo "[qk-lut-wait-parallel] cifar100_pid=$C100_PID"
echo "[qk-lut-wait-parallel] stage2_pid=$STAGE2_PID"
wait "$C100_PID" "$STAGE2_PID"
echo "[qk-lut-wait-parallel] done=$(date -Is)"
