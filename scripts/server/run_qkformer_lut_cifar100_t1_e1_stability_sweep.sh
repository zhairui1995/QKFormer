#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
export QKFORMER_LUT_GPU="${QKFORMER_LUT_GPU:-2}"
qk_lut_configure_gpu

if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-lut-c100-e1-stability] missing python3/python"
  exit 1
fi

if [[ -z "${QKFORMER_LUT_CKPT:-}" ]]; then
  QKFORMER_LUT_CKPT="$("$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dirs = [p for p in root.glob("results/qkformer_cifar100_train_*") if (p / "checkpoint_manifest.json").exists()]
if not dirs:
    raise SystemExit("no completed CIFAR-100 training result found")
latest = max(dirs, key=lambda p: (p.stat().st_mtime, str(p)))
manifest = json.loads((latest / "checkpoint_manifest.json").read_text(encoding="utf-8"))
checkpoint = manifest.get("best_checkpoint") or manifest.get("latest_checkpoint")
if not checkpoint or not Path(checkpoint).exists():
    raise SystemExit(f"missing checkpoint from {latest}")
print(Path(checkpoint).resolve())
PY
)"
  export QKFORMER_LUT_CKPT
fi

SEED_SWEEP="${QKFORMER_LUT_E1_SEED_SWEEP:-42,43,44}"
export QKFORMER_LUT_E1_CONFIG=configs/qkformer_lut_cifar100_t1_e1_recon.yaml
export QKFORMER_LUT_TIME_STEP=1
export QKFORMER_LUT_DATA_DIR="${QKFORMER_LUT_CIFAR100_DATA_DIR:-$ROOT/data/cifar100}"
export QKFORMER_LUT_E1_CALIB_BATCHES="${QKFORMER_LUT_E1_CALIB_BATCHES:-128}"
export QKFORMER_LUT_E1_EVAL_BATCHES="${QKFORMER_LUT_E1_EVAL_BATCHES:-0}"
export QKFORMER_LUT_E1_CALIB_SHUFFLE=1

echo "[qk-lut-c100-e1-stability] checkpoint=$QKFORMER_LUT_CKPT"
echo "[qk-lut-c100-e1-stability] seeds=$SEED_SWEEP"
echo "[qk-lut-c100-e1-stability] calib_batches=$QKFORMER_LUT_E1_CALIB_BATCHES"
echo "[qk-lut-c100-e1-stability] eval_batches=$QKFORMER_LUT_E1_EVAL_BATCHES"

IFS=',' read -r -a SEEDS <<< "$SEED_SWEEP"
for seed in "${SEEDS[@]}"; do
  seed="$(echo "$seed" | xargs)"
  [[ -n "$seed" ]] || continue
  echo "[qk-lut-c100-e1-stability] run_seed=$seed"
  QKFORMER_LUT_E1_SEED="$seed" \
    bash scripts/server/run_qkformer_lut_e1_recon.sh --gpu "$QKFORMER_LUT_GPU"
done

echo "[qk-lut-c100-e1-stability] done=$(date -Is)"
