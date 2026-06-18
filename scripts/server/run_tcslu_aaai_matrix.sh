#!/usr/bin/env bash
set -euo pipefail

ROOT="${TCSLU_ROOT:-/home/lbz/mac_agent/sdr-lutattn-qkformer-lut}"
PYTHON="${TCSLU_PYTHON:-/home/lbz/miniconda/envs/sdr/bin/python}"
DATA_DIR="${TCSLU_DATA_DIR:-/home/datasets}"
GPU="${1:-${TCSLU_GPU:-0}}"
CASE="${2:-${TCSLU_CASE:-qkf_c10_t1_status}}"
TS="$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_qkformer_probe() {
  local family="$1"
  local time_step="$2"
  local checkpoint="$3"
  local tag="$4"
  local max_eval_batches="$5"
  local result_dir="$ROOT/results/tcslu_${tag}_${TS}"
  mkdir -p "$result_dir"
  echo "[tcslu] qkformer_probe family=$family T=$time_step result_dir=$result_dir"
  local args=(
    "$ROOT/tools/qkformer_lut_current_unit_probe.py"
    --root "$ROOT" \
    --family "$family" \
    --data-dir "$DATA_DIR" \
    --checkpoint "$checkpoint" \
    --result-dir "$result_dir" \
    --dim 384 \
    --layer 4 \
    --time-step "$time_step" \
    --targets stage1.0.tssa \
    --batch-size "${TCSLU_BATCH_SIZE:-128}" \
    --workers "${TCSLU_WORKERS:-4}" \
    --calib-batches "${TCSLU_CALIB_BATCHES:-128}" \
    --min-support "${TCSLU_MIN_SUPPORT:-2}" \
    --seed "${TCSLU_SEED:-42}" \
    --local-cuda-index 0 \
    --alphas "${TCSLU_ALPHAS:-0.0,1.0}"
  )
  if [[ -n "$max_eval_batches" ]]; then
    args+=(--max-eval-batches "$max_eval_batches")
  fi
  "$PYTHON" "${args[@]}"
}

latest_cifar100_checkpoint() {
  "$PYTHON" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
candidates = sorted(
    [p for p in root.glob("results/qkformer_cifar100_train_*") if (p / "checkpoint_manifest.json").exists()],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not candidates:
    raise SystemExit("no qkformer_cifar100_train_* checkpoint_manifest.json found")
for path in candidates:
    manifest = json.load(open(path / "checkpoint_manifest.json"))
    ckpt = manifest.get("best_checkpoint")
    if ckpt:
        print(ckpt)
        break
else:
    raise SystemExit("no best_checkpoint found")
PY
}

case "$CASE" in
  qkf_c10_t1_status)
    run_qkformer_probe \
      cifar10 \
      1 \
      "$ROOT/results/qkformer_cifar10_train_20260606_001151/output/qkformer_cifar10_t1/model_best.pth.tar" \
      qkf_c10_t1_status \
      "${TCSLU_MAX_EVAL_BATCHES:-16}"
    ;;
  qkf_c100_t1_full)
    CKPT="${TCSLU_CKPT:-$(latest_cifar100_checkpoint)}"
    run_qkformer_probe cifar100 1 "$CKPT" qkf_c100_t1_full "${TCSLU_MAX_EVAL_BATCHES:-100000}"
    ;;
  qkf_c100_t4_full)
    CKPT="${TCSLU_CKPT:-$(latest_cifar100_checkpoint)}"
    run_qkformer_probe cifar100 4 "$CKPT" qkf_c100_t4_full "${TCSLU_MAX_EVAL_BATCHES:-100000}"
    ;;
  spik_c10_t4_boundary_rerun)
    : "${TCSLU_SPIKFORMER_REPO:?set TCSLU_SPIKFORMER_REPO to external spikformer repo}"
    : "${TCSLU_SPIKFORMER_CKPT:?set TCSLU_SPIKFORMER_CKPT to trained Spikformer checkpoint}"
    RESULT_DIR="$ROOT/results/tcslu_spik_c10_t4_boundary_rerun_${TS}"
    mkdir -p "$RESULT_DIR"
    "$PYTHON" "$ROOT/tools/run_spikformer_current_injection_diagnostic.py" \
      --repo-root "$TCSLU_SPIKFORMER_REPO" \
      --data-root "$DATA_DIR" \
      --checkpoint "$TCSLU_SPIKFORMER_CKPT" \
      --result-dir "$RESULT_DIR" \
      --batch-size "${TCSLU_BATCH_SIZE:-256}" \
      --workers "${TCSLU_WORKERS:-8}" \
      --calib-batches "${TCSLU_CALIB_BATCHES:-128}" \
      --train-batches "${TCSLU_TRAIN_BATCHES:-196}" \
      --min-support "${TCSLU_MIN_SUPPORT:-2}" \
      --seed "${TCSLU_SEED:-42}" \
      --local-cuda-index 0 \
      --alphas "${TCSLU_ALPHAS:-0.0,1.0}"
    ;;
  *)
    echo "usage: $0 [GPU] [qkf_c10_t1_status|qkf_c100_t1_full|qkf_c100_t4_full|spik_c10_t4_boundary_rerun]" >&2
    exit 2
    ;;
esac
