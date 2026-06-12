#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
export QKFORMER_LUT_GPU="${QKFORMER_LUT_GPU:-2}"
qk_lut_configure_gpu

if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-lut-c100-t4-oracle] missing python3/python" >&2
  exit 1
fi

QKFORMER_LUT_CKPT="$("$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.glob("results/qkformer_cifar100_train_*"):
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.exists():
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("time_step")) != "4":
        continue
    checkpoint = manifest.get("best_checkpoint") or manifest.get("latest_checkpoint")
    if checkpoint and Path(checkpoint).exists():
        candidates.append((path.stat().st_mtime, checkpoint))
if not candidates:
    raise SystemExit("no completed CIFAR-100 T=4 checkpoint found")
print(max(candidates)[1])
PY
)"

export QKFORMER_LUT_CKPT
export QKFORMER_LUT_TIME_STEP=4
export QKFORMER_LUT_DATA_DIR="${QKFORMER_LUT_CIFAR100_DATA_DIR:-$ROOT/data/cifar100}"
export QKFORMER_LUT_E3_CONFIG=configs/qkformer_lut_cifar100_t1_e3_trainable_lut.yaml
export QKFORMER_LUT_E3_ALPHA_INIT="${QKFORMER_LUT_E3_ALPHA_INIT:-0.025}"
export QKFORMER_LUT_E3_LEARN_ALPHA=0
export QKFORMER_LUT_E3_SWEEP_TARGETS="${QKFORMER_LUT_E3_SWEEP_TARGETS:-stage1.0.tssa}"
export QKFORMER_LUT_E3_MODE_SWEEP="${QKFORMER_LUT_E3_MODE_SWEEP:-global_plus_address_lut,global_plus_shuffled_address_lut,global_mean}"
export QKFORMER_LUT_E3_SEED_SWEEP="${QKFORMER_LUT_E3_SEED_SWEEP:-42,43,44}"
export QKFORMER_LUT_E3_EPOCHS="${QKFORMER_LUT_E3_EPOCHS:-2}"
export QKFORMER_LUT_E3_CALIB_BATCHES="${QKFORMER_LUT_E3_CALIB_BATCHES:-128}"
export QKFORMER_LUT_E3_TRAIN_BATCHES="${QKFORMER_LUT_E3_TRAIN_BATCHES:-128}"
export QKFORMER_LUT_E3_EVAL_BATCHES="${QKFORMER_LUT_E3_EVAL_BATCHES:-0}"
export QKFORMER_LUT_E3_EVAL_BATCH_SIZE="${QKFORMER_LUT_E3_EVAL_BATCH_SIZE:-64}"
export QKFORMER_LUT_E3_EVAL_AMP=1
export QKFORMER_LUT_E3_EVAL_LOADER=timm
export QKFORMER_LUT_E3_SAVE_PER_SAMPLE=1

echo "[qk-lut-c100-t4-oracle] checkpoint=$QKFORMER_LUT_CKPT"
echo "[qk-lut-c100-t4-oracle] gpu=$QKFORMER_LUT_GPU"
echo "[qk-lut-c100-t4-oracle] targets=$QKFORMER_LUT_E3_SWEEP_TARGETS"
echo "[qk-lut-c100-t4-oracle] modes=$QKFORMER_LUT_E3_MODE_SWEEP"
echo "[qk-lut-c100-t4-oracle] seeds=$QKFORMER_LUT_E3_SEED_SWEEP"
echo "[qk-lut-c100-t4-oracle] alpha=$QKFORMER_LUT_E3_ALPHA_INIT"
echo "[qk-lut-c100-t4-oracle] calib_batches=$QKFORMER_LUT_E3_CALIB_BATCHES"
echo "[qk-lut-c100-t4-oracle] train_batches=$QKFORMER_LUT_E3_TRAIN_BATCHES"
echo "[qk-lut-c100-t4-oracle] eval_batches=$QKFORMER_LUT_E3_EVAL_BATCHES"
echo "[qk-lut-c100-t4-oracle] save_per_sample=$QKFORMER_LUT_E3_SAVE_PER_SAMPLE"

bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh --gpu "$QKFORMER_LUT_GPU"
"$PYTHON_BIN" scripts/local/analyze_qk_lut_selective_oracle.py
bash scripts/server/package_qkformer_lut_cifar100_t4_selective_oracle.sh

echo "[qk-lut-c100-t4-oracle] done=$(date -Is)"
