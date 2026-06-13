#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
export QKFORMER_LUT_GPU="${QKFORMER_LUT_GPU:-2}"
qk_lut_configure_gpu

if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-lut-c100-t4-gate] missing python3/python" >&2
  exit 1
fi

QKFORMER_LUT_CKPT="$("$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.glob("results/qkformer_cifar100_train_*"):
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.exists():
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("time_step")) != "4":
        continue
    checkpoint = manifest.get("best_checkpoint") or manifest.get("latest_checkpoint")
    if checkpoint and Path(checkpoint).exists():
        candidates.append((path.stat().st_mtime, checkpoint))
if not candidates:
    raise SystemExit("no completed CIFAR-100 T=4 checkpoint found")
print(max(candidates)[1])
PY
)"

export QKFORMER_LUT_CKPT
export QKFORMER_LUT_TIME_STEP=4
export QKFORMER_LUT_DATA_DIR="${QKFORMER_LUT_CIFAR100_DATA_DIR:-$ROOT/data/cifar100}"
export QKFORMER_LUT_E3_CONFIG=configs/qkformer_lut_cifar100_t1_e3_trainable_lut.yaml
export QKFORMER_LUT_E3_ALPHA_INIT=0.025
export QKFORMER_LUT_E3_LEARN_ALPHA=0
export QKFORMER_LUT_E3_SWEEP_TARGETS=stage1.0.tssa
export QKFORMER_LUT_E3_MODE_SWEEP="${QKFORMER_LUT_E3_MODE_SWEEP:-global_plus_address_lut,global_plus_shuffled_address_lut,global_mean}"
export QKFORMER_LUT_E3_SEED_SWEEP="${QKFORMER_LUT_E3_SEED_SWEEP:-42,43,44}"
export QKFORMER_LUT_E3_EPOCHS=2
export QKFORMER_LUT_E3_CALIB_BATCHES=128
export QKFORMER_LUT_E3_TRAIN_BATCHES=128
export QKFORMER_LUT_E3_GATE_BATCHES=128
export QKFORMER_LUT_E3_GATE_ENABLED=1
export QKFORMER_LUT_E3_GATE_SEED=45
export QKFORMER_LUT_E3_EVAL_BATCHES=0
export QKFORMER_LUT_E3_EVAL_BATCH_SIZE=64
export QKFORMER_LUT_E3_EVAL_AMP=1
export QKFORMER_LUT_E3_EVAL_LOADER=timm
export QKFORMER_LUT_E3_SAVE_PER_SAMPLE=1
export QKFORMER_LUT_E3_SAVE_GATE_PER_SAMPLE=1

echo "[qk-lut-c100-t4-gate] checkpoint=$QKFORMER_LUT_CKPT"
echo "[qk-lut-c100-t4-gate] gpu=$QKFORMER_LUT_GPU"
echo "[qk-lut-c100-t4-gate] modes=$QKFORMER_LUT_E3_MODE_SWEEP"
echo "[qk-lut-c100-t4-gate] seeds=$QKFORMER_LUT_E3_SEED_SWEEP"
echo "[qk-lut-c100-t4-gate] gate_seed=$QKFORMER_LUT_E3_GATE_SEED"
echo "[qk-lut-c100-t4-gate] partition=calib[0,16000),train[16000,32000),gate[32000,50000)"

bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh --gpu "$QKFORMER_LUT_GPU"

if [[ "${QKFORMER_LUT_GATE_ANALYZE_AFTER:-1}" == "1" ]]; then
  "$PYTHON_BIN" scripts/local/analyze_qk_lut_deterministic_gate.py
  bash scripts/server/package_qkformer_lut_cifar100_t4_deterministic_gate.sh
fi

echo "[qk-lut-c100-t4-gate] done=$(date -Is)"
