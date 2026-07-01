#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu
if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[native-qklut-lif-spikformer-c10] missing python3/python"
  exit 1
fi

MODE="${NATIVE_QKLUT_LIF_SPIKFORMER_RUN_MODE:-smoke}"
TARGET_SCOPE="${NATIVE_QKLUT_LIF_TARGET_SCOPE:-attention}"
SPIKFORMER_REPO="${NATIVE_QKLUT_LIF_SPIKFORMER_REPO:-/home/lbz/mac_agent/external/spikformer}"
DATA_ROOT="${QKFORMER_LUT_DATA_DIR:-/home/datasets}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$ROOT/results/native_qklut_lif_spikformer_qk_contract_${TARGET_SCOPE}_c10_t4_seed42_${MODE}_${STAMP}"
mkdir -p "$RESULT_DIR"

case "$MODE" in
  smoke)
    EPOCHS="${NATIVE_QKLUT_LIF_SPIKFORMER_EPOCHS:-1}"
    MAX_TRAIN_BATCHES="${NATIVE_QKLUT_LIF_SPIKFORMER_MAX_TRAIN_BATCHES:-8}"
    MAX_VAL_BATCHES="${NATIVE_QKLUT_LIF_SPIKFORMER_MAX_VAL_BATCHES:-4}"
    BATCH_SIZE="${NATIVE_QKLUT_LIF_SPIKFORMER_BATCH_SIZE:-16}"
    VAL_BATCH_SIZE="${NATIVE_QKLUT_LIF_SPIKFORMER_VAL_BATCH_SIZE:-16}"
    WORKERS="${NATIVE_QKLUT_LIF_SPIKFORMER_WORKERS:-2}"
    ;;
  pilot)
    EPOCHS="${NATIVE_QKLUT_LIF_SPIKFORMER_EPOCHS:-30}"
    MAX_TRAIN_BATCHES="${NATIVE_QKLUT_LIF_SPIKFORMER_MAX_TRAIN_BATCHES:-0}"
    MAX_VAL_BATCHES="${NATIVE_QKLUT_LIF_SPIKFORMER_MAX_VAL_BATCHES:-0}"
    BATCH_SIZE="${NATIVE_QKLUT_LIF_SPIKFORMER_BATCH_SIZE:-64}"
    VAL_BATCH_SIZE="${NATIVE_QKLUT_LIF_SPIKFORMER_VAL_BATCH_SIZE:-128}"
    WORKERS="${NATIVE_QKLUT_LIF_SPIKFORMER_WORKERS:-8}"
    ;;
  *)
    echo "[native-qklut-lif-spikformer-c10] unsupported mode=$MODE"
    exit 2
    ;;
esac

export PYTHONPATH="$ROOT:$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
export QKFORMER_TRAIN_SEED="${QKFORMER_TRAIN_SEED:-42}"
LOG_FILE="$RESULT_DIR/launcher.log"
CODE_COMMIT="${NATIVE_QKLUT_LIF_CODE_COMMIT:-$(git rev-parse --short HEAD)}"

run_row() {
  local row="$1"
  local extra_args=()
  local row_dir="$RESULT_DIR/$row"
  mkdir -p "$row_dir"
  if [[ "$row" == native_qklut_lif_* ]]; then
    extra_args=(
      --qklut-lif-native
      --qklut-lif-target-scope "$TARGET_SCOPE"
      --qklut-lif-state-bits 6
      --qklut-lif-input-bits 8
      --qklut-lif-x-range -8.0 8.0
      --qklut-lif-v-range 0.0 2.0
    )
  fi
  echo "[native-qklut-lif-spikformer-c10] row=$row start=$(date -Is)"
  "$PYTHON_BIN" "$ROOT/tools/train_cross_arch_cifar.py" \
    --track spikformer \
    --repo-root "$SPIKFORMER_REPO" \
    --data-root "$DATA_ROOT" \
    --dataset cifar10 \
    --result-dir "$row_dir" \
    --spikformer-attn-mode qk_sum \
    --spikformer-attn-scope all \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --workers "$WORKERS" \
    --lr "${NATIVE_QKLUT_LIF_SPIKFORMER_LR:-5e-4}" \
    --weight-decay "${NATIVE_QKLUT_LIF_SPIKFORMER_WD:-0.05}" \
    --seed "$QKFORMER_TRAIN_SEED" \
    --amp \
    --max-train-batches "$MAX_TRAIN_BATCHES" \
    --max-val-batches "$MAX_VAL_BATCHES" \
    "${extra_args[@]}"
  echo "[native-qklut-lif-spikformer-c10] row=$row done=$(date -Is)"
}

{
  echo "[native-qklut-lif-spikformer-c10] root=$ROOT"
  echo "[native-qklut-lif-spikformer-c10] result_dir=$RESULT_DIR"
  echo "[native-qklut-lif-spikformer-c10] mode=$MODE"
  echo "[native-qklut-lif-spikformer-c10] target_scope=$TARGET_SCOPE"
  echo "[native-qklut-lif-spikformer-c10] commit=$CODE_COMMIT"
  echo "[native-qklut-lif-spikformer-c10] python=$PYTHON_BIN"
  echo "[native-qklut-lif-spikformer-c10] data_root=$DATA_ROOT"
  echo "[native-qklut-lif-spikformer-c10] spikformer_repo=$SPIKFORMER_REPO"
  echo "[native-qklut-lif-spikformer-c10] epochs=$EPOCHS train_batches=$MAX_TRAIN_BATCHES val_batches=$MAX_VAL_BATCHES"
  echo "[native-qklut-lif-spikformer-c10] batch_size=$BATCH_SIZE val_batch_size=$VAL_BATCH_SIZE workers=$WORKERS seed=$QKFORMER_TRAIN_SEED"
  qk_lut_log_gpu "$PYTHON_BIN" "native-qklut-lif-spikformer-c10"
  run_row baseline_spikformer_qk_contract
  run_row "native_qklut_lif_spikformer_${TARGET_SCOPE}"
  "$PYTHON_BIN" - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1]).resolve()
rows = []
for manifest_path in sorted(result_dir.glob("*/checkpoint_manifest.json")):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows.append({
        "row": manifest_path.parent.name,
        "manifest": str(manifest_path),
        "best_epoch": int(manifest["best_epoch"]),
        "best_top1": float(manifest["best_top1"]),
        "native_qklut_lif_summary": manifest.get("native_qklut_lif_summary"),
    })
(result_dir / "comparison.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"result_dir": str(result_dir), "rows": rows}, sort_keys=True))
PY
  echo "[native-qklut-lif-spikformer-c10] done=$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
