#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
export QKFORMER_LUT_GPU="${QKFORMER_LUT_GPU:-2}"
qk_lut_configure_gpu

if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-lut-c100-e4] missing python3/python"
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

export QKFORMER_LUT_E3_CONFIG=configs/qkformer_lut_cifar100_t1_e3_trainable_lut.yaml
export QKFORMER_LUT_DATA_DIR="${QKFORMER_LUT_CIFAR100_DATA_DIR:-$ROOT/data/cifar100}"
export QKFORMER_LUT_TIME_STEP=1
export QKFORMER_LUT_E3_SWEEP_TARGETS=stage1.0.tssa
export QKFORMER_LUT_E3_SEED_SWEEP="${QKFORMER_LUT_E4_ADAPTER_SEEDS:-42,43,44}"
export QKFORMER_LUT_E3_MODE_SWEEP="${QKFORMER_LUT_E4_MODE_SWEEP:-address_lut,global_mean,factorized_sum_lut,factorized_gated_lut,matched_param_address_lut,factorized_shuffled_lut}"
export QKFORMER_LUT_E3_ALPHA_INIT="${QKFORMER_LUT_E4_ALPHA_INIT:-0.025}"
export QKFORMER_LUT_E3_LEARN_ALPHA=0
export QKFORMER_LUT_E3_EPOCHS="${QKFORMER_LUT_E4_EPOCHS:-2}"
export QKFORMER_LUT_E3_CALIB_BATCHES="${QKFORMER_LUT_E4_CALIB_BATCHES:-128}"
export QKFORMER_LUT_E3_TRAIN_BATCHES="${QKFORMER_LUT_E4_TRAIN_BATCHES:-128}"
export QKFORMER_LUT_E3_EVAL_BATCHES=0

echo "[qk-lut-c100-e4] checkpoint=$QKFORMER_LUT_CKPT"
echo "[qk-lut-c100-e4] modes=$QKFORMER_LUT_E3_MODE_SWEEP"
echo "[qk-lut-c100-e4] seeds=$QKFORMER_LUT_E3_SEED_SWEEP"
bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh --gpu "$QKFORMER_LUT_GPU"
"$PYTHON_BIN" scripts/local/analyze_qk_lut_e4_gate.py

