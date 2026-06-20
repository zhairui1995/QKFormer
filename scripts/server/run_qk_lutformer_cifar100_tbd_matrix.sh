#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${QKFORMER_PYTHON:-${PYTHON:-$HOME/.conda_envs/sdr/bin/python}}"
DATA_DIR="${QKFORMER_LUT_CIFAR100_DATA_DIR:-$ROOT/data/cifar100}"
GPU="${QK_LUTFORMER_GPU:-0}"
CASE="${1:-smoke}"
TS="$(date +%Y%m%d_%H%M%S)"
MATRIX_DIR="${QK_LUTFORMER_RESULT_DIR:-$ROOT/results/qk_lutformer_cifar100_${CASE}_${TS}}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$MATRIX_DIR/runs"

latest_checkpoint() {
  "$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for result_dir in sorted(root.glob("results/qkformer_cifar100_train_*"), reverse=True):
    manifest_path = result_dir / "checkpoint_manifest.json"
    if not manifest_path.exists():
        continue
    checkpoint = json.loads(manifest_path.read_text()).get("best_checkpoint")
    if checkpoint and Path(checkpoint).exists():
        print(checkpoint)
        raise SystemExit(0)
raise SystemExit("no usable CIFAR-100 checkpoint found; set QK_LUTFORMER_CKPT")
PY
}

CHECKPOINT="${QK_LUTFORMER_CKPT:-$(latest_checkpoint)}"
CALIB_BATCHES="${QK_LUTFORMER_CALIB_BATCHES:-128}"
EVAL_BATCHES="${QK_LUTFORMER_EVAL_BATCHES:-100000}"
BATCH_SIZE="${QK_LUTFORMER_BATCH_SIZE:-128}"
WORKERS="${QK_LUTFORMER_WORKERS:-4}"

run_probe() {
  local run_id="$1"
  local lut_mode="$2"
  local seed="$3"
  local min_support="$4"
  local corruption="$5"
  local severity="$6"
  local run_dir="$MATRIX_DIR/runs/$run_id"
  mkdir -p "$run_dir"
  "$PYTHON_BIN" - "$run_dir/run_metadata.json" "$CASE" "$run_id" "$lut_mode" "$seed" "$min_support" "$corruption" "$severity" <<'PY'
import json
import sys
from pathlib import Path

path, case, run_id, lut_mode, seed, min_support, corruption, severity = sys.argv[1:]
Path(path).write_text(json.dumps({
    "case": case,
    "run_id": run_id,
    "lut_mode": lut_mode,
    "seed": int(seed),
    "min_support": int(min_support),
    "corruption": corruption,
    "corruption_severity": int(severity),
}, indent=2, sort_keys=True) + "\n")
PY
  echo "[qk-lutformer-c100] case=$CASE run=$run_id mode=$lut_mode seed=$seed support=$min_support corruption=$corruption/$severity"
  "$PYTHON_BIN" "$ROOT/tools/qkformer_lut_current_unit_probe.py" \
    --root "$ROOT" \
    --family cifar100 \
    --data-dir "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" \
    --result-dir "$run_dir" \
    --dim 384 \
    --layer 4 \
    --time-step "${QK_LUTFORMER_TIME_STEP:-1}" \
    --targets stage1.0.tssa \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --calib-batches "$CALIB_BATCHES" \
    --max-eval-batches "$EVAL_BATCHES" \
    --min-support "$min_support" \
    --seed "$seed" \
    --local-cuda-index 0 \
    --alphas 0.0,1.0 \
    --lut-mode "$lut_mode" \
    --corruption "$corruption" \
    --corruption-severity "$severity"
}

case "$CASE" in
  smoke|component)
    run_probe aligned_seed42 aligned_lut 42 2 none 1
    ;;
  semantic_controls)
    run_probe aligned aligned_lut 42 2 none 1
    run_probe shuffled shuffled_address 42 2 none 1
    run_probe token_channel token_channel_mean 42 2 none 1
    run_probe global global_mean 42 2 none 1
    run_probe same_size_random random_table 42 2 none 1
    ;;
  calibration_subsets)
    run_probe calib_seed42 aligned_lut 42 2 none 1
    run_probe calib_seed43 aligned_lut 43 2 none 1
    run_probe calib_seed44 aligned_lut 44 2 none 1
    ;;
  support_thresholds)
    run_probe support1 aligned_lut 42 1 none 1
    run_probe support2 aligned_lut 42 2 none 1
    run_probe support4 aligned_lut 42 4 none 1
    ;;
  mild_corruption)
    run_probe clean aligned_lut 42 2 none 1
    run_probe noise1 aligned_lut 42 2 noise 1
    run_probe noise2 aligned_lut 42 2 noise 2
    run_probe brightness1 aligned_lut 42 2 brightness 1
    run_probe brightness2 aligned_lut 42 2 brightness 2
    ;;
  *)
    echo "usage: $0 [smoke|component|semantic_controls|calibration_subsets|support_thresholds|mild_corruption]" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" "$ROOT/scripts/local/collect_qk_lutformer_cifar100_matrix.py" "$MATRIX_DIR"
echo "[qk-lutformer-c100] result_dir=$MATRIX_DIR"
