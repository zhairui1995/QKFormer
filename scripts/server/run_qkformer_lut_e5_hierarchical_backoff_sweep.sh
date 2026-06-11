#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
export QKFORMER_LUT_GPU="${QKFORMER_LUT_GPU:-2}"
qk_lut_configure_gpu

if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-lut-e5] missing python3/python"
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

export QKFORMER_LUT_E1_CONFIG="${QKFORMER_LUT_E5_CONFIG:-configs/qkformer_lut_cifar100_t1_e1_recon.yaml}"
export QKFORMER_LUT_DATA_DIR="${QKFORMER_LUT_CIFAR100_DATA_DIR:-$ROOT/data/cifar100}"
export QKFORMER_LUT_TIME_STEP=1
export QKFORMER_LUT_E1_CALIB_SHUFFLE="${QKFORMER_LUT_E5_CALIB_SHUFFLE:-1}"
export QKFORMER_LUT_E1_EVAL_BATCHES="${QKFORMER_LUT_E5_EVAL_BATCHES:-0}"

CALIB_SIZES="${QKFORMER_LUT_E5_CALIB_SIZES:-1,2,4,8,32,128,512}"
SEEDS="${QKFORMER_LUT_E5_SEEDS:-42,43,44}"
TAG_PREFIX="${QKFORMER_LUT_E5_TAG_PREFIX:-e5_hier}"
ARCHIVE_NAME="${QKFORMER_LUT_E5_ARCHIVE_NAME:-qk_lutformer_e5_hierarchical_backoff_artifacts.tar.gz}"

echo "[qk-lut-e5] root=$ROOT"
echo "[qk-lut-e5] commit=$(git rev-parse --short HEAD)"
echo "[qk-lut-e5] checkpoint=$QKFORMER_LUT_CKPT"
echo "[qk-lut-e5] config=$QKFORMER_LUT_E1_CONFIG"
echo "[qk-lut-e5] data_dir=$QKFORMER_LUT_DATA_DIR"
echo "[qk-lut-e5] gpu=$QKFORMER_LUT_GPU"
echo "[qk-lut-e5] calib_sizes=$CALIB_SIZES"
echo "[qk-lut-e5] seeds=$SEEDS"
echo "[qk-lut-e5] eval_batches=$QKFORMER_LUT_E1_EVAL_BATCHES"
echo "[qk-lut-e5] tag_prefix=$TAG_PREFIX"

IFS=',' read -r -a SIZE_GROUPS <<< "$CALIB_SIZES"
IFS=',' read -r -a SEED_GROUPS <<< "$SEEDS"

for size in "${SIZE_GROUPS[@]}"; do
  size="$(echo "$size" | xargs)"
  [[ -n "$size" ]] || continue
  for seed in "${SEED_GROUPS[@]}"; do
    seed="$(echo "$seed" | xargs)"
    [[ -n "$seed" ]] || continue
    tag="${TAG_PREFIX}_calib${size}_seed${seed}"
    if ls "results"/qkformer_lut_e1_recon_*_"$tag"/metrics.json >/dev/null 2>&1; then
      echo "[qk-lut-e5] skip_existing tag=$tag"
      continue
    fi
    echo "[qk-lut-e5] run calib_batches=$size seed=$seed tag=$tag"
    QKFORMER_LUT_E1_CALIB_BATCHES="$size" \
    QKFORMER_LUT_E1_SEED="$seed" \
    QKFORMER_LUT_RESULT_TAG="$tag" \
      bash scripts/server/run_qkformer_lut_e1_recon.sh
  done
done

if [[ "$TAG_PREFIX" == "e5_hier" ]]; then
  "$PYTHON_BIN" scripts/local/analyze_qk_lut_e5_hierarchical_backoff.py
fi

mapfile -t ARTIFACTS < <(
  for d in results/qkformer_lut_e1_recon_*_"$TAG_PREFIX"_calib*_seed*; do
    [[ -d "$d" ]] || continue
    for f in metrics.json train_log.txt module_reconstruction.csv; do
      [[ -f "$d/$f" ]] && echo "$d/$f"
    done
  done
  [[ -f results/qk_lutformer_e5_hierarchical_backoff.csv ]] && echo results/qk_lutformer_e5_hierarchical_backoff.csv
  [[ -f results/qk_lutformer_e5_hierarchical_backoff.json ]] && echo results/qk_lutformer_e5_hierarchical_backoff.json
  [[ -f results/qk_lutformer_e5_hierarchical_backoff_report.md ]] && echo results/qk_lutformer_e5_hierarchical_backoff_report.md
)

if [[ "${#ARTIFACTS[@]}" -gt 0 ]]; then
  tar -czf "$ARCHIVE_NAME" "${ARTIFACTS[@]}"
  echo "[qk-lut-e5] archive=$ROOT/$ARCHIVE_NAME"
fi
echo "[qk-lut-e5] done=$(date -Is)"
