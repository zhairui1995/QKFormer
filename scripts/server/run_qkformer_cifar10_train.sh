#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu
if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-train] missing python3/python"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$ROOT/results/qkformer_cifar10_train_${TS}"
LOG_FILE="$RESULT_DIR/train_log.txt"
mkdir -p "$RESULT_DIR"

export QKFORMER_LUT_DATA_DIR="${QKFORMER_LUT_DATA_DIR:-$ROOT/data/cifar10}"
EPOCHS="${QKFORMER_TRAIN_EPOCHS:-400}"
BATCH_SIZE="${QKFORMER_TRAIN_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${QKFORMER_VAL_BATCH_SIZE:-64}"
WORKERS="${QKFORMER_TRAIN_WORKERS:-8}"

{
  echo "[qk-train] root=$ROOT"
  echo "[qk-train] result_dir=$RESULT_DIR"
  echo "[qk-train] start=$(date -Is)"
  echo "[qk-train] commit=$(git rev-parse --short HEAD)"
  echo "[qk-train] python=$PYTHON_BIN"
  echo "[qk-train] data_dir=$QKFORMER_LUT_DATA_DIR"
  echo "[qk-train] epochs=$EPOCHS batch_size=$BATCH_SIZE val_batch_size=$VAL_BATCH_SIZE workers=$WORKERS"

  bash scripts/server/install_qkformer_lut_deps.sh
  bash scripts/server/link_cifar10_data.sh
  qk_lut_log_gpu "$PYTHON_BIN" "qk-train"

  cd "$ROOT/cifar10"
  "$PYTHON_BIN" train.py \
    -c cifar10.yml \
    --model QKFormer \
    -data-dir "$QKFORMER_LUT_DATA_DIR" \
    --output "$RESULT_DIR/output" \
    --experiment qkformer_cifar10 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --workers "$WORKERS"

  cd "$ROOT"
  find "$RESULT_DIR" \( -name "*.pth" -o -name "*.pth.tar" \) -print | sort > "$RESULT_DIR/checkpoint_manifest.txt"
  "$PYTHON_BIN" - "$RESULT_DIR" <<'PY'
import json
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
    "result_dir": str(result_dir),
    "summary_csv": str(next(result_dir.rglob("summary.csv"), "")),
    "checkpoints": [str(path) for path in checkpoints],
    "best_checkpoint": str(newest(best_candidates) or newest(checkpoints) or ""),
    "latest_checkpoint": str(newest(latest_candidates) or newest(checkpoints) or ""),
}
(result_dir / "checkpoint_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"[qk-train] checkpoint_manifest_json={result_dir / 'checkpoint_manifest.json'}")
print(f"[qk-train] best_checkpoint={manifest['best_checkpoint']}")
print(f"[qk-train] latest_checkpoint={manifest['latest_checkpoint']}")
PY
  echo "[qk-train] checkpoints:"
  cat "$RESULT_DIR/checkpoint_manifest.txt"
  echo "[qk-train] done=$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
