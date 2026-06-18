#!/usr/bin/env bash
set -euo pipefail

ROOT="${QK_GROUPED_ROOT:-/home/lbz/mac_agent/sdr-lutattn-qkformer-lut}"
PYTHON="${QK_GROUPED_PYTHON:-/home/lbz/miniconda/envs/sdr/bin/python}"
DATA_DIR="${QK_GROUPED_DATA_DIR:-/home/datasets}"
GPU="${1:-0}"
TIME_STEP="${2:-1}"
GROUP_SIZE="${3:-2}"

if [[ "$TIME_STEP" != "1" && "$TIME_STEP" != "4" ]]; then
  echo "usage: $0 [GPU] [1|4] [GROUP_SIZE]" >&2
  exit 2
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="$("$PYTHON" - "$ROOT" "$TIME_STEP" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
time_step = int(sys.argv[2])
candidates = []
for manifest_path in root.glob("results/qkformer_cifar100_train_*/checkpoint_manifest.json"):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = manifest.get("best_checkpoint")
    if not checkpoint or not Path(checkpoint).exists():
        continue
    if f"_t{time_step}_" not in str(checkpoint):
        continue
    if "seed42" not in str(checkpoint):
        continue
    candidates.append((manifest_path.stat().st_mtime, checkpoint))
if not candidates:
    raise SystemExit(f"missing CIFAR-100 T={time_step} seed-42 checkpoint")
print(max(candidates)[1])
PY
)"

TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$ROOT/results/qk_grouped_projection_g${GROUP_SIZE}_c100_t${TIME_STEP}_all4_full_${TS}"
mkdir -p "$RESULT_DIR"

"$PYTHON" tools/qkformer_lut_current_unit_probe.py \
  --root "$ROOT" \
  --family cifar100 \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --result-dir "$RESULT_DIR" \
  --dim 384 \
  --layer 4 \
  --time-step "$TIME_STEP" \
  --targets stage1.0.tssa,stage2.0.tssa,stage3.0.ssa,stage3.1.ssa \
  --batch-size "${QK_GROUPED_BATCH_SIZE:-128}" \
  --workers "${QK_GROUPED_WORKERS:-4}" \
  --max-eval-batches "${QK_GROUPED_MAX_EVAL_BATCHES:-100000}" \
  --seed 42 \
  --local-cuda-index 0 \
  --projection-group-size "$GROUP_SIZE" \
  --group-lut-only

echo "[qk-grouped-all4] result_dir=$RESULT_DIR"
