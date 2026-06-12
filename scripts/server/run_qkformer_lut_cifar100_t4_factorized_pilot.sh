#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
export QKFORMER_LUT_GPU="${QKFORMER_LUT_GPU:-2}"
qk_lut_configure_gpu

if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-lut-c100-t4-factorized] missing python3/python" >&2
  exit 1
fi

QKFORMER_LUT_CKPT="$($PYTHON_BIN - "$ROOT" <<'PY'
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
export QKFORMER_LUT_E3_ALPHA_INIT=0.025
export QKFORMER_LUT_E3_LEARN_ALPHA=0
export QKFORMER_LUT_E3_SWEEP_TARGETS=stage1.0.tssa
export QKFORMER_LUT_E3_MODE_SWEEP=factorized_sum_lut,factorized_gated_lut,matched_param_address_lut,factorized_shuffled_lut
export QKFORMER_LUT_E3_SEED_SWEEP=42,43,44
export QKFORMER_LUT_E3_EPOCHS=2
export QKFORMER_LUT_E3_CALIB_BATCHES=128
export QKFORMER_LUT_E3_TRAIN_BATCHES=128
export QKFORMER_LUT_E3_EVAL_BATCHES=0
export QKFORMER_LUT_E3_EVAL_BATCH_SIZE=64
export QKFORMER_LUT_E3_EVAL_AMP=1
export QKFORMER_LUT_E3_EVAL_LOADER=timm

echo "[qk-lut-c100-t4-factorized] checkpoint=$QKFORMER_LUT_CKPT"
echo "[qk-lut-c100-t4-factorized] gpu=$QKFORMER_LUT_GPU"
echo "[qk-lut-c100-t4-factorized] protocol_version=4"
echo "[qk-lut-c100-t4-factorized] modes=$QKFORMER_LUT_E3_MODE_SWEEP"
echo "[qk-lut-c100-t4-factorized] seeds=$QKFORMER_LUT_E3_SEED_SWEEP"

bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh --gpu "$QKFORMER_LUT_GPU"
"$PYTHON_BIN" scripts/local/analyze_qk_lut_cifar100_t4_factorized.py

mapfile -t RESULT_DIRS < <(ls -td results/qkformer_lut_e3_trainable_lut_* | head -12)
FILES=(
  results/qk_lutformer_cifar100_t4_factorized.json
  results/qk_lutformer_cifar100_t4_factorized.csv
  results/qk_lutformer_cifar100_t4_factorized.md
)
for dir in "${RESULT_DIRS[@]}"; do
  FILES+=("$dir/metrics.json" "$dir/train_log.txt")
done
tar -czf qk_lutformer_cifar100_t4_factorized_artifacts.tar.gz "${FILES[@]}"

echo "[qk-lut-c100-t4-factorized] artifact=$ROOT/qk_lutformer_cifar100_t4_factorized_artifacts.tar.gz"
echo "[qk-lut-c100-t4-factorized] done=$(date -Is)"
