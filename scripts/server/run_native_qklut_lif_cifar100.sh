#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu
if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[native-qklut-lif-c100] missing python3/python"
  exit 1
fi

MODE="${NATIVE_QKLUT_LIF_RUN_MODE:-smoke}"
DATA_DIR="${QKFORMER_LUT_CIFAR100_DATA_DIR:-${QKFORMER_LUT_DATA_DIR:-$ROOT/data/cifar100}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$ROOT/results/native_qklut_lif_attention_c100_t4_seed42_${MODE}_${STAMP}"
mkdir -p "$RESULT_DIR"

case "$MODE" in
  smoke)
    EPOCHS="${NATIVE_QKLUT_LIF_EPOCHS:-1}"
    MAX_TRAIN_BATCHES="${NATIVE_QKLUT_LIF_MAX_TRAIN_BATCHES:-8}"
    MAX_EVAL_BATCHES="${NATIVE_QKLUT_LIF_MAX_EVAL_BATCHES:-4}"
    BATCH_SIZE="${NATIVE_QKLUT_LIF_BATCH_SIZE:-16}"
    VAL_BATCH_SIZE="${NATIVE_QKLUT_LIF_VAL_BATCH_SIZE:-16}"
    WORKERS="${NATIVE_QKLUT_LIF_WORKERS:-2}"
    ;;
  pilot)
    EPOCHS="${NATIVE_QKLUT_LIF_EPOCHS:-30}"
    MAX_TRAIN_BATCHES="${NATIVE_QKLUT_LIF_MAX_TRAIN_BATCHES:-0}"
    MAX_EVAL_BATCHES="${NATIVE_QKLUT_LIF_MAX_EVAL_BATCHES:-0}"
    BATCH_SIZE="${NATIVE_QKLUT_LIF_BATCH_SIZE:-64}"
    VAL_BATCH_SIZE="${NATIVE_QKLUT_LIF_VAL_BATCH_SIZE:-64}"
    WORKERS="${NATIVE_QKLUT_LIF_WORKERS:-8}"
    ;;
  full)
    EPOCHS="${NATIVE_QKLUT_LIF_EPOCHS:-400}"
    MAX_TRAIN_BATCHES="${NATIVE_QKLUT_LIF_MAX_TRAIN_BATCHES:-0}"
    MAX_EVAL_BATCHES="${NATIVE_QKLUT_LIF_MAX_EVAL_BATCHES:-0}"
    BATCH_SIZE="${NATIVE_QKLUT_LIF_BATCH_SIZE:-64}"
    VAL_BATCH_SIZE="${NATIVE_QKLUT_LIF_VAL_BATCH_SIZE:-64}"
    WORKERS="${NATIVE_QKLUT_LIF_WORKERS:-8}"
    ;;
  *)
    echo "[native-qklut-lif-c100] unsupported mode=$MODE"
    exit 2
    ;;
esac

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export QKFORMER_LUT_TIME_STEP=4
export QKFORMER_TRAIN_SEED="${QKFORMER_TRAIN_SEED:-42}"
LOG_FILE="$RESULT_DIR/launcher.log"

run_row() {
  local row="$1"
  local extra_args=()
  local experiment="qkformer_c100_t4_seed${QKFORMER_TRAIN_SEED}_${row}_${MODE}"
  if [[ "$row" == "native_qklut_lif_attention" ]]; then
    extra_args=(
      --qklut-lif-native
      --qklut-lif-target-scope attention
      --qklut-lif-state-bits 6
      --qklut-lif-input-bits 8
      --qklut-lif-x-range -8.0 8.0
      --qklut-lif-v-range 0.0 2.0
    )
  fi
  echo "[native-qklut-lif-c100] row=$row start=$(date -Is)"
  (
    cd "$ROOT/cifar100"
    "$PYTHON_BIN" train.py \
      -c cifar100.yml \
      --model QKFormer \
      -data-dir "$DATA_DIR" \
      --output "$RESULT_DIR/output" \
      --experiment "$experiment" \
      --epochs "$EPOCHS" \
      --time-step 4 \
      --seed "$QKFORMER_TRAIN_SEED" \
      --batch-size "$BATCH_SIZE" \
      --val-batch-size "$VAL_BATCH_SIZE" \
      --workers "$WORKERS" \
      --max-train-batches "$MAX_TRAIN_BATCHES" \
      --max-eval-batches "$MAX_EVAL_BATCHES" \
      --cooldown-epochs 0 \
      --checkpoint-hist 1 \
      "${extra_args[@]}"
  )
  echo "[native-qklut-lif-c100] row=$row done=$(date -Is)"
}

{
  echo "[native-qklut-lif-c100] root=$ROOT"
  echo "[native-qklut-lif-c100] result_dir=$RESULT_DIR"
  echo "[native-qklut-lif-c100] mode=$MODE"
  echo "[native-qklut-lif-c100] commit=$(git rev-parse --short HEAD)"
  echo "[native-qklut-lif-c100] python=$PYTHON_BIN"
  echo "[native-qklut-lif-c100] data_dir=$DATA_DIR"
  echo "[native-qklut-lif-c100] epochs=$EPOCHS train_batches=$MAX_TRAIN_BATCHES eval_batches=$MAX_EVAL_BATCHES"
  echo "[native-qklut-lif-c100] batch_size=$BATCH_SIZE val_batch_size=$VAL_BATCH_SIZE workers=$WORKERS seed=$QKFORMER_TRAIN_SEED"
  CIFAR100_SOURCE_ROOT="${CIFAR100_SOURCE_ROOT:-$DATA_DIR}" \
    QKFORMER_LUT_CIFAR100_DATA_DIR="$DATA_DIR" \
    PYTHON="$PYTHON_BIN" \
    bash scripts/server/link_cifar100_data.sh
  qk_lut_log_gpu "$PYTHON_BIN" "native-qklut-lif-c100"
  run_row baseline_qkformer
  run_row native_qklut_lif_attention
  "$PYTHON_BIN" - "$RESULT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1]).resolve()
rows = []
for summary in sorted(result_dir.glob("output/*/summary.csv")):
    with summary.open(newline="", encoding="utf-8") as handle:
        data = [
            row for row in csv.DictReader(handle)
            if row.get("epoch", "").strip().lstrip("-").isdigit()
        ]
    if not data:
        raise RuntimeError(f"empty or malformed summary: {summary}")
    best = max(data, key=lambda row: float(row.get("eval_top1", row.get("top1", "nan"))))
    rows.append({
        "row": summary.parent.name,
        "summary_csv": str(summary),
        "best_epoch": int(best["epoch"]),
        "best_top1": float(best.get("eval_top1", best.get("top1"))),
        "best_top5": float(best.get("eval_top5", best.get("top5"))),
        "best_loss": float(best.get("eval_loss", best.get("loss"))),
    })
(result_dir / "comparison.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"result_dir": str(result_dir), "rows": rows}, sort_keys=True))
PY
  echo "[native-qklut-lif-c100] done=$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
