#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
qk_lut_configure_gpu
if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-lut-e2-align] missing python3/python"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$ROOT/results/qkformer_lut_e2_alignment_test_${TS}"
LOG_FILE="$RESULT_DIR/train_log.txt"
mkdir -p "$RESULT_DIR"

export QKFORMER_LUT_TIME_STEP="${QKFORMER_LUT_TIME_STEP:-1}"
export QKFORMER_LUT_E2_TARGETS="${QKFORMER_LUT_E2_TARGETS:-stage1.0.tssa}"
export QKFORMER_LUT_E2_CALIB_BATCHES="${QKFORMER_LUT_E2_CALIB_BATCHES:-0}"
export QKFORMER_LUT_E2_EVAL_BATCHES="${QKFORMER_LUT_E2_EVAL_BATCHES:-0}"
export QKFORMER_LUT_E2_BLEND="${QKFORMER_LUT_E2_BLEND:-1.0}"
export QKFORMER_LUT_E2_ALIGNMENT_MODES="${QKFORMER_LUT_E2_ALIGNMENT_MODES:-address_lut,shuffled_address_lut,global_mean}"
export QKFORMER_LUT_E2_MODE_SEED="${QKFORMER_LUT_E2_MODE_SEED:-42}"

if [[ -z "${QKFORMER_LUT_DATA_DIR:-}" && -d "$ROOT/data/cifar10/cifar-10-batches-py" ]]; then
  export QKFORMER_LUT_DATA_DIR="$ROOT/data/cifar10"
fi

if [[ -z "${QKFORMER_LUT_CKPT:-}" ]]; then
  QKFORMER_LUT_CKPT="$("$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.glob("results/qkformer_cifar10_train_*"):
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.exists():
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("time_step")) != "1":
        continue
    checkpoint = manifest.get("best_checkpoint") or ""
    if checkpoint and Path(checkpoint).exists():
        summary_csv = manifest.get("summary_csv") or ""
        best_top1 = -1.0
        if summary_csv and Path(summary_csv).exists():
            with Path(summary_csv).open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        best_top1 = max(best_top1, float(row.get("eval_top1") or -1.0))
                    except ValueError:
                        pass
        candidates.append((best_top1, path, checkpoint))
if not candidates:
    raise SystemExit("no T=1 checkpoint_manifest with an existing best_checkpoint found")
_best_top1, path, checkpoint = max(candidates, key=lambda item: (item[0], item[1].stat().st_mtime, str(item[1])))
print(checkpoint)
PY
)"
  export QKFORMER_LUT_CKPT
fi

{
  echo "[qk-lut-e2-align] root=$ROOT"
  echo "[qk-lut-e2-align] result_dir=$RESULT_DIR"
  echo "[qk-lut-e2-align] start=$(date -Is)"
  echo "[qk-lut-e2-align] commit=$(git rev-parse --short HEAD)"
  echo "[qk-lut-e2-align] checkpoint=$QKFORMER_LUT_CKPT"
  echo "[qk-lut-e2-align] time_step=$QKFORMER_LUT_TIME_STEP"
  echo "[qk-lut-e2-align] targets=$QKFORMER_LUT_E2_TARGETS"
  echo "[qk-lut-e2-align] calib_batches=$QKFORMER_LUT_E2_CALIB_BATCHES"
  echo "[qk-lut-e2-align] eval_batches=$QKFORMER_LUT_E2_EVAL_BATCHES"
  echo "[qk-lut-e2-align] blend=$QKFORMER_LUT_E2_BLEND"
  echo "[qk-lut-e2-align] modes=$QKFORMER_LUT_E2_ALIGNMENT_MODES"
  qk_lut_log_gpu "$PYTHON_BIN" "qk-lut-e2-align"

  "$PYTHON_BIN" tools/qkformer_lut_e2_alignment_test.py \
    --config configs/qkformer_lut_e2_replace.yaml \
    --output-dir "$RESULT_DIR"

  "$PYTHON_BIN" - "$RESULT_DIR/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"[qk-lut-e2-align] metrics_ok={sys.argv[1]}")
for mode, result in metrics["mode_results"].items():
    cls = result["classification"]
    print(
        f"[qk-lut-e2-align] mode={mode} "
        f"baseline_top1={cls['baseline']['top1']} "
        f"replacement_top1={cls['replacement']['top1']} "
        f"delta_top1={cls['delta']['top1']} "
        f"kl={cls['replacement']['kl_to_baseline']} "
        f"logit_mse={cls['replacement']['logit_mse']}"
    )
PY
  echo "[qk-lut-e2-align] done=$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
