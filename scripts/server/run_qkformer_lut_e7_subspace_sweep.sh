#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
export QKFORMER_LUT_GPU="${QKFORMER_LUT_GPU:-2}"
qk_lut_configure_gpu

if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-lut-e7-subspace] missing python3/python"
  exit 1
fi

if [[ -z "${QKFORMER_LUT_CKPT:-}" ]]; then
  QKFORMER_LUT_CKPT="$("$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dirs = [path for path in root.glob("results/qkformer_cifar100_train_*") if (path / "checkpoint_manifest.json").exists()]
if not dirs:
    raise SystemExit("no completed CIFAR-100 training result found")
latest = max(dirs, key=lambda path: (path.stat().st_mtime, str(path)))
manifest = json.loads((latest / "checkpoint_manifest.json").read_text(encoding="utf-8"))
checkpoint = manifest.get("best_checkpoint") or manifest.get("latest_checkpoint")
if not checkpoint or not Path(checkpoint).exists():
    raise SystemExit(f"missing checkpoint from {latest}")
print(Path(checkpoint).resolve())
PY
)"
  export QKFORMER_LUT_CKPT
fi

export QKFORMER_LUT_E1_CONFIG="${QKFORMER_LUT_E7_CONFIG:-configs/qkformer_lut_cifar100_t1_e1_recon.yaml}"
export QKFORMER_LUT_DATA_DIR="${QKFORMER_LUT_CIFAR100_DATA_DIR:-$ROOT/data/cifar100}"
export QKFORMER_LUT_TIME_STEP="${QKFORMER_LUT_E7_TIME_STEP:-1}"
export QKFORMER_LUT_E1_CALIB_SHUFFLE="${QKFORMER_LUT_E7_CALIB_SHUFFLE:-1}"
export QKFORMER_LUT_E1_EVAL_BATCHES="${QKFORMER_LUT_E7_EVAL_BATCHES:-0}"
unset QKFORMER_LUT_E1_HIERARCHY_BUDGET_FRACTION

CALIB_SIZES="${QKFORMER_LUT_E7_CALIB_SIZES:-8,32,128,512}"
SEEDS="${QKFORMER_LUT_E7_SEEDS:-42,43,44}"
MIN_COUNTS="${QKFORMER_LUT_E7_MIN_COUNTS:-2}"
ARCHIVE_NAME="${QKFORMER_LUT_E7_ARCHIVE_NAME:-qk_lutformer_e7_subspace_artifacts.tar.gz}"

echo "[qk-lut-e7-subspace] root=$ROOT"
echo "[qk-lut-e7-subspace] commit=$(git rev-parse --short HEAD)"
echo "[qk-lut-e7-subspace] checkpoint=$QKFORMER_LUT_CKPT"
echo "[qk-lut-e7-subspace] config=$QKFORMER_LUT_E1_CONFIG"
echo "[qk-lut-e7-subspace] data_dir=$QKFORMER_LUT_DATA_DIR"
echo "[qk-lut-e7-subspace] time_step=$QKFORMER_LUT_TIME_STEP"
echo "[qk-lut-e7-subspace] gpu=$QKFORMER_LUT_GPU"
echo "[qk-lut-e7-subspace] calib_sizes=$CALIB_SIZES"
echo "[qk-lut-e7-subspace] seeds=$SEEDS"
echo "[qk-lut-e7-subspace] min_counts=$MIN_COUNTS"
echo "[qk-lut-e7-subspace] eval_batches=$QKFORMER_LUT_E1_EVAL_BATCHES"

IFS=',' read -r -a SIZE_GROUPS <<< "$CALIB_SIZES"
IFS=',' read -r -a SEED_GROUPS <<< "$SEEDS"
IFS=',' read -r -a MIN_GROUPS <<< "$MIN_COUNTS"

for size in "${SIZE_GROUPS[@]}"; do
  size="$(echo "$size" | xargs)"
  [[ -n "$size" ]] || continue
  for seed in "${SEED_GROUPS[@]}"; do
    seed="$(echo "$seed" | xargs)"
    [[ -n "$seed" ]] || continue
    for min_count in "${MIN_GROUPS[@]}"; do
      min_count="$(echo "$min_count" | xargs)"
      [[ -n "$min_count" ]] || continue
      tag="e7_subspace_calib${size}_seed${seed}_min${min_count}"
      if ls "results"/qkformer_lut_e1_recon_*_"$tag"/metrics.json >/dev/null 2>&1; then
        echo "[qk-lut-e7-subspace] skip_existing tag=$tag"
        continue
      fi
      echo "[qk-lut-e7-subspace] run calib_batches=$size seed=$seed min_count=$min_count tag=$tag"
      QKFORMER_LUT_E1_CALIB_BATCHES="$size" \
      QKFORMER_LUT_E1_SEED="$seed" \
      QKFORMER_LUT_E1_MIN_COUNT="$min_count" \
      QKFORMER_LUT_RESULT_TAG="$tag" \
        bash scripts/server/run_qkformer_lut_e1_recon.sh
    done
  done
done

"$PYTHON_BIN" scripts/local/analyze_qk_lut_e7_subspace.py

mapfile -t ARTIFACTS < <(
  for d in results/qkformer_lut_e1_recon_*_e7_subspace_calib*_seed*_min*; do
    [[ -d "$d" ]] || continue
    for f in metrics.json train_log.txt module_reconstruction.csv; do
      [[ -f "$d/$f" ]] && echo "$d/$f"
    done
  done
  [[ -f results/qk_lutformer_e7_subspace.csv ]] && echo results/qk_lutformer_e7_subspace.csv
  [[ -f results/qk_lutformer_e7_subspace.json ]] && echo results/qk_lutformer_e7_subspace.json
  [[ -f results/qk_lutformer_e7_subspace_report.md ]] && echo results/qk_lutformer_e7_subspace_report.md
)

if [[ "${#ARTIFACTS[@]}" -gt 0 ]]; then
  tar -czf "$ARCHIVE_NAME" "${ARTIFACTS[@]}"
  echo "[qk-lut-e7-subspace] archive=$ROOT/$ARCHIVE_NAME"
fi
echo "[qk-lut-e7-subspace] done=$(date -Is)"
