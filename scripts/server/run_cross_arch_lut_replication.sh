#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lbz/mac_agent/sdr-lutattn-qkformer-lut"
PYTHON="/home/lbz/miniconda/envs/sdr/bin/python"
DATA_ROOT="/home/datasets"
EXTERNAL_ROOT="/home/lbz/mac_agent/external"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TRACK GPU_ID"
  echo "  TRACK: spikformer or sew_resnet34"
  exit 2
fi
TRACK="$1"
GPU_ID="$2"
TS="$(date +%Y%m%d_%H%M%S)"

case "$TRACK" in
  spikformer)
    REPO_ROOT="$EXTERNAL_ROOT/spikformer"
    TRAIN_EPOCHS_C10="${CROSS_ARCH_SPIKFORMER_C10_EPOCHS:-300}"
    TRAIN_EPOCHS_C100="${CROSS_ARCH_SPIKFORMER_C100_EPOCHS:-300}"
    BATCH_SIZE="${CROSS_ARCH_SPIKFORMER_BATCH_SIZE:-128}"
    VAL_BATCH_SIZE="${CROSS_ARCH_SPIKFORMER_VAL_BATCH_SIZE:-256}"
    ;;
  sew_resnet34)
    REPO_ROOT="$EXTERNAL_ROOT/Spike-Element-Wise-ResNet"
    TRAIN_EPOCHS_C10="${CROSS_ARCH_SEW_C10_EPOCHS:-300}"
    TRAIN_EPOCHS_C100="${CROSS_ARCH_SEW_C100_EPOCHS:-300}"
    BATCH_SIZE="${CROSS_ARCH_SEW_BATCH_SIZE:-128}"
    VAL_BATCH_SIZE="${CROSS_ARCH_SEW_VAL_BATCH_SIZE:-256}"
    ;;
  *)
    echo "[cross-arch] unsupported track=$TRACK" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUN_ROOT="$ROOT/results/cross_arch_${TRACK}_${TS}"
LOG_FILE="$RUN_ROOT/launcher.log"
mkdir -p "$RUN_ROOT"

{
  echo "[cross-arch] host=$(hostname)"
  echo "[cross-arch] start=$(date -Is)"
  echo "[cross-arch] root=$ROOT"
  echo "[cross-arch] run_root=$RUN_ROOT"
  echo "[cross-arch] track=$TRACK"
  echo "[cross-arch] repo_root=$REPO_ROOT"
  echo "[cross-arch] data_root=$DATA_ROOT"
  echo "[cross-arch] python=$PYTHON"
  echo "[cross-arch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "$PYTHON" - <<'PY'
import sys, torch, torchvision
print(f"[cross-arch] sys_executable={sys.executable}")
print(f"[cross-arch] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[cross-arch] device_count={torch.cuda.device_count()} current={torch.cuda.current_device()} name={torch.cuda.get_device_name(0)}")
print(f"[cross-arch] torchvision={torchvision.__version__}")
PY

  echo "[cross-arch] === phase1 train cifar10 ==="
  C10_TRAIN="$RUN_ROOT/cifar10_train"
  "$PYTHON" "$ROOT/tools/train_cross_arch_cifar.py" \
    --track "$TRACK" \
    --repo-root "$REPO_ROOT" \
    --data-root "$DATA_ROOT" \
    --dataset cifar10 \
    --result-dir "$C10_TRAIN" \
    --epochs "$TRAIN_EPOCHS_C10" \
    --batch-size "$BATCH_SIZE" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --workers "${CROSS_ARCH_WORKERS:-8}" \
    --lr "${CROSS_ARCH_LR:-0.0005}" \
    --weight-decay "${CROSS_ARCH_WEIGHT_DECAY:-0.05}" \
    --seed "${CROSS_ARCH_SEED:-42}" \
    --amp

  C10_CKPT="$("$PYTHON" - "$C10_TRAIN/checkpoint_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["best_checkpoint"])
PY
)"
  echo "[cross-arch] c10_best_checkpoint=$C10_CKPT"

  echo "[cross-arch] === phase1 lut cifar10 ==="
  C10_LUT="$RUN_ROOT/cifar10_lut_phase1"
  "$PYTHON" "$ROOT/tools/run_cross_arch_lut_phase1.py" \
    --track "$TRACK" \
    --repo-root "$REPO_ROOT" \
    --data-root "$DATA_ROOT" \
    --dataset cifar10 \
    --checkpoint "$C10_CKPT" \
    --result-dir "$C10_LUT" \
    --batch-size "$VAL_BATCH_SIZE" \
    --workers "${CROSS_ARCH_WORKERS:-8}" \
    --calib-batches "${CROSS_ARCH_CALIB_BATCHES:-8}" \
    --min-support "${CROSS_ARCH_MIN_SUPPORT:-2}" \
    --max-drop-pct 5.0 \
    --seed "${CROSS_ARCH_SEED:-42}" \
    --local-cuda-index 0

  PHASE1_VERDICT="$("$PYTHON" - "$C10_LUT/metrics.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["verdict"])
PY
)"
  echo "[cross-arch] phase1_verdict=$PHASE1_VERDICT"
  if [[ "$PHASE1_VERDICT" != "PASS" ]]; then
    echo "[cross-arch] stop_before_phase2=phase1_gate_failed"
    exit 0
  fi

  echo "[cross-arch] === phase2 train cifar100 ==="
  C100_TRAIN="$RUN_ROOT/cifar100_train"
  "$PYTHON" "$ROOT/tools/train_cross_arch_cifar.py" \
    --track "$TRACK" \
    --repo-root "$REPO_ROOT" \
    --data-root "$DATA_ROOT" \
    --dataset cifar100 \
    --result-dir "$C100_TRAIN" \
    --epochs "$TRAIN_EPOCHS_C100" \
    --batch-size "$BATCH_SIZE" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --workers "${CROSS_ARCH_WORKERS:-8}" \
    --lr "${CROSS_ARCH_LR:-0.0005}" \
    --weight-decay "${CROSS_ARCH_WEIGHT_DECAY:-0.05}" \
    --seed "${CROSS_ARCH_SEED:-42}" \
    --amp

  C100_CKPT="$("$PYTHON" - "$C100_TRAIN/checkpoint_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["best_checkpoint"])
PY
)"
  echo "[cross-arch] c100_best_checkpoint=$C100_CKPT"

  echo "[cross-arch] === phase2 lut cifar100 ==="
  C100_LUT="$RUN_ROOT/cifar100_lut_phase2"
  "$PYTHON" "$ROOT/tools/run_cross_arch_lut_phase1.py" \
    --track "$TRACK" \
    --repo-root "$REPO_ROOT" \
    --data-root "$DATA_ROOT" \
    --dataset cifar100 \
    --checkpoint "$C100_CKPT" \
    --result-dir "$C100_LUT" \
    --batch-size "$VAL_BATCH_SIZE" \
    --workers "${CROSS_ARCH_WORKERS:-8}" \
    --calib-batches "${CROSS_ARCH_CALIB_BATCHES:-8}" \
    --min-support "${CROSS_ARCH_MIN_SUPPORT:-2}" \
    --max-drop-pct 5.0 \
    --seed "${CROSS_ARCH_SEED:-42}" \
    --local-cuda-index 0

  echo "[cross-arch] done=$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
