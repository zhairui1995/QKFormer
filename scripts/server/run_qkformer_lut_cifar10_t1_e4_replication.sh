#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
export QKFORMER_LUT_GPU="${QKFORMER_LUT_GPU:-2}"
qk_lut_configure_gpu
if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-lut-c10-e4] missing python3/python"
  exit 1
fi

if [[ "${QKFORMER_LUT_E4_FORCE_C10:-0}" != "1" ]]; then
  "$PYTHON_BIN" - "$ROOT/results/qk_lutformer_e4_gate.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit("missing E4 gate report; run the CIFAR-100 pilot first")
report = json.loads(path.read_text(encoding="utf-8"))
passing = [item["candidate"] for item in report.get("candidates", []) if item.get("pilot_pass")]
if not passing:
    raise SystemExit("CIFAR-100 E4 pilot did not pass; refusing CIFAR-10 replication")
print(f"[qk-lut-c10-e4] pilot_pass={','.join(passing)}")
PY
fi

export QKFORMER_LUT_E3_CONFIG=configs/qkformer_lut_e3_trainable_lut.yaml
export QKFORMER_LUT_DATA_DIR="${QKFORMER_LUT_CIFAR10_DATA_DIR:-$ROOT/data/cifar10}"
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

for checkpoint_seed in 43 44; do
  QKFORMER_LUT_CKPT="$("$PYTHON_BIN" - "$ROOT" "$checkpoint_seed" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = sys.argv[2]
candidates = []
for train_dir in root.glob("results/qkformer_cifar10_train_*"):
    manifest_path = train_dir / "checkpoint_manifest.json"
    if not manifest_path.exists():
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("time_step")) != "1" or str(manifest.get("seed")) != seed:
        continue
    checkpoint = manifest.get("best_checkpoint") or manifest.get("latest_checkpoint")
    if checkpoint and Path(checkpoint).exists():
        candidates.append(Path(checkpoint))
if not candidates:
    raise SystemExit(f"no T=1 CIFAR-10 checkpoint found for seed {seed}")
print(max(candidates, key=lambda path: (path.stat().st_mtime, str(path))).resolve())
PY
)"
  export QKFORMER_LUT_CKPT
  echo "[qk-lut-c10-e4] checkpoint_seed=$checkpoint_seed checkpoint=$QKFORMER_LUT_CKPT"
  bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh --gpu "$QKFORMER_LUT_GPU"
done

"$PYTHON_BIN" scripts/local/analyze_qk_lut_e4_gate.py
