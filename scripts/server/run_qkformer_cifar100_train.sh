#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu
if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-c100-train] missing python3/python"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$ROOT/results/qkformer_cifar100_train_${TS}"
LOG_FILE="$RESULT_DIR/train_log.txt"
mkdir -p "$RESULT_DIR"

export QKFORMER_LUT_CIFAR100_DATA_DIR="${QKFORMER_LUT_CIFAR100_DATA_DIR:-$ROOT/data/cifar100}"
EPOCHS="${QKFORMER_CIFAR100_TRAIN_EPOCHS:-400}"
BATCH_SIZE="${QKFORMER_CIFAR100_TRAIN_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${QKFORMER_CIFAR100_VAL_BATCH_SIZE:-64}"
WORKERS="${QKFORMER_CIFAR100_TRAIN_WORKERS:-8}"
TIME_STEP="${QKFORMER_LUT_TIME_STEP:-1}"
SEED="${QKFORMER_CIFAR100_TRAIN_SEED:-42}"
EXPERIMENT="${QKFORMER_CIFAR100_TRAIN_EXPERIMENT:-qkformer_cifar100_t${TIME_STEP}_seed${SEED}}"

{
  echo "[qk-c100-train] root=$ROOT"
  echo "[qk-c100-train] result_dir=$RESULT_DIR"
  echo "[qk-c100-train] start=$(date -Is)"
  echo "[qk-c100-train] commit=$(git rev-parse --short HEAD)"
  echo "[qk-c100-train] python=$PYTHON_BIN"
  echo "[qk-c100-train] data_dir=$QKFORMER_LUT_CIFAR100_DATA_DIR"
  echo "[qk-c100-train] epochs=$EPOCHS batch_size=$BATCH_SIZE val_batch_size=$VAL_BATCH_SIZE workers=$WORKERS time_step=$TIME_STEP seed=$SEED experiment=$EXPERIMENT"

  bash scripts/server/install_qkformer_lut_deps.sh
  bash scripts/server/link_cifar100_data.sh
  qk_lut_log_gpu "$PYTHON_BIN" "qk-c100-train"

  cd "$ROOT/cifar100"
  "$PYTHON_BIN" train.py \
    -c cifar100.yml \
    --model QKFormer \
    -data-dir "$QKFORMER_LUT_CIFAR100_DATA_DIR" \
    --output "$RESULT_DIR/output" \
    --experiment "$EXPERIMENT" \
    --epochs "$EPOCHS" \
    --time-step "$TIME_STEP" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --workers "$WORKERS"

  cd "$ROOT"
  find "$RESULT_DIR" \( -name "*.pth" -o -name "*.pth.tar" \) -print | sort > "$RESULT_DIR/checkpoint_manifest.txt"
  "$PYTHON_BIN" - "$RESULT_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

result_dir = Path(sys.argv[1]).resolve()
checkpoints = sorted(
    [
        path.resolve()
        for pattern in ("*.pth", "*.pth.tar")
        for path in result_dir.rglob(pattern)
    ],
    key=lambda path: str(path),
)

def newest(paths):
    if not paths:
        return None
    return max(paths, key=lambda path: (path.stat().st_mtime, str(path)))

best_candidates = [
    path for path in checkpoints
    if "best" in path.name.lower() or "best" in str(path.parent).lower()
]
latest_candidates = [
    path for path in checkpoints
    if "last" in path.name.lower() or "latest" in path.name.lower()
]

manifest = {
    "dataset": "cifar100",
    "result_dir": str(result_dir),
    "summary_csv": str(next(result_dir.rglob("summary.csv"), "")),
    "time_step": os.environ.get("QKFORMER_LUT_TIME_STEP", "1"),
    "seed": os.environ.get("QKFORMER_CIFAR100_TRAIN_SEED", "42"),
    "checkpoints": [str(path) for path in checkpoints],
    "best_checkpoint": str(newest(best_candidates) or newest(checkpoints) or ""),
    "latest_checkpoint": str(newest(latest_candidates) or newest(checkpoints) or ""),
}
(result_dir / "checkpoint_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"[qk-c100-train] checkpoint_manifest_json={result_dir / 'checkpoint_manifest.json'}")
print(f"[qk-c100-train] best_checkpoint={manifest['best_checkpoint']}")
print(f"[qk-c100-train] latest_checkpoint={manifest['latest_checkpoint']}")
PY
  echo "[qk-c100-train] checkpoints:"
  cat "$RESULT_DIR/checkpoint_manifest.txt"
  echo "[qk-c100-train] done=$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
