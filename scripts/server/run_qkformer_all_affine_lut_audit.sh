#!/usr/bin/env bash
set -euo pipefail

ROOT="${QK_AFFINE_ROOT:-/home/lbz/mac_agent/sdr-lutattn-qkformer-lut}"
PYTHON="${QK_AFFINE_PYTHON:-/home/lbz/miniconda/envs/sdr/bin/python}"
DATA_DIR="${QK_AFFINE_DATA_DIR:-/home/datasets}"
GPU="${1:-0}"
CATEGORY="${2:-qkv}"
ATTEMPT="${3:-main}"
TIME_STEP="${4:-1}"

case "$CATEGORY" in
  qkv|mlp|patch|classifier|projection|all_affine) ;;
  *) echo "invalid category: $CATEGORY" >&2; exit 2 ;;
esac
case "$ATTEMPT" in
  main) INTEGER_MAX=7; UNIFORM_BITS=8 ;;
  retry) INTEGER_MAX=15; UNIFORM_BITS=10 ;;
  *) echo "invalid attempt: $ATTEMPT" >&2; exit 2 ;;
esac
case "$TIME_STEP" in
  1) EXPECTED_CLEAN=77.78 ;;
  4) EXPECTED_CLEAN=81.09 ;;
  *) echo "invalid time step: $TIME_STEP" >&2; exit 2 ;;
esac

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
    if f"_t{time_step}_" not in str(checkpoint) or "seed42" not in str(checkpoint):
        continue
    candidates.append((manifest_path.stat().st_mtime, checkpoint))
if not candidates:
    raise SystemExit(f"missing CIFAR-100 T={time_step} seed-42 checkpoint")
print(max(candidates)[1])
PY
)"

TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$ROOT/results/qk_all_affine_${CATEGORY}_${ATTEMPT}_c100_t${TIME_STEP}_${TS}"
mkdir -p "$RESULT_DIR"

"$PYTHON" tools/qkformer_all_affine_lut_probe.py \
  --root "$ROOT" \
  --family cifar100 \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --result-dir "$RESULT_DIR" \
  --category "$CATEGORY" \
  --dim 384 \
  --layer 4 \
  --time-step "$TIME_STEP" \
  --batch-size "${QK_AFFINE_BATCH_SIZE:-32}" \
  --workers "${QK_AFFINE_WORKERS:-4}" \
  --calib-batches "${QK_AFFINE_CALIB_BATCHES:-32}" \
  --max-eval-batches "${QK_AFFINE_MAX_EVAL_BATCHES:-100000}" \
  --integer-max "$INTEGER_MAX" \
  --uniform-bits "$UNIFORM_BITS" \
  --input-chunk "${QK_AFFINE_INPUT_CHUNK:-4}" \
  --max-drop 0.25 \
  --max-nrmse 0.01 \
  --max-clip-rate 0.0001 \
  --expected-clean-top1 "$EXPECTED_CLEAN" \
  --clean-tolerance 0.15 \
  --seed 42 \
  --local-cuda-index 0 \
  2>&1 | tee "$RESULT_DIR/run.log"

echo "[qk-all-affine] result_dir=$RESULT_DIR"
