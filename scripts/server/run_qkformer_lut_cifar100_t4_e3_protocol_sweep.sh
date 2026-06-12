#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/server/qkformer_lut_common.sh"
qk_lut_parse_gpu_args "$@"
export QKFORMER_LUT_GPU="${QKFORMER_LUT_GPU:-2}"
qk_lut_configure_gpu

if ! PYTHON_BIN="$(qk_lut_select_python)"; then
  echo "[qk-lut-c100-t4-e3-protocol] missing python3/python" >&2
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
export QKFORMER_LUT_E3_MODE_SWEEP=address_lut,global_mean,token_channel_lut,shuffled_address_lut
export QKFORMER_LUT_E3_SEED_SWEEP=42,43,44
export QKFORMER_LUT_E3_EPOCHS=2
export QKFORMER_LUT_E3_CALIB_BATCHES=128
export QKFORMER_LUT_E3_TRAIN_BATCHES=128
export QKFORMER_LUT_E3_EVAL_BATCHES=0
export QKFORMER_LUT_E3_EVAL_BATCH_SIZE=64
export QKFORMER_LUT_E3_EVAL_AMP=1
export QKFORMER_LUT_E3_EVAL_LOADER=timm

echo "[qk-lut-c100-t4-e3-protocol] checkpoint=$QKFORMER_LUT_CKPT"
echo "[qk-lut-c100-t4-e3-protocol] gpu=$QKFORMER_LUT_GPU"
echo "[qk-lut-c100-t4-e3-protocol] protocol_version=4"
echo "[qk-lut-c100-t4-e3-protocol] eval_batch_size=$QKFORMER_LUT_E3_EVAL_BATCH_SIZE"
echo "[qk-lut-c100-t4-e3-protocol] eval_amp=$QKFORMER_LUT_E3_EVAL_AMP"
echo "[qk-lut-c100-t4-e3-protocol] eval_loader=$QKFORMER_LUT_E3_EVAL_LOADER"

echo "[qk-lut-c100-t4-e3-protocol] identity_preflight_start"
QKFORMER_LUT_E3_MODE=address_lut \
QKFORMER_LUT_E3_SEED=9001 \
QKFORMER_LUT_E3_TARGETS=stage1.0.tssa \
QKFORMER_LUT_E3_ALPHA_INIT=0 \
QKFORMER_LUT_E3_EPOCHS=0 \
QKFORMER_LUT_E3_CALIB_BATCHES=1 \
QKFORMER_LUT_E3_TRAIN_BATCHES=1 \
  bash scripts/server/run_qkformer_lut_e3_trainable_lut.sh --gpu "$QKFORMER_LUT_GPU"

"$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
result_dir = max(root.glob("results/qkformer_lut_e3_trainable_lut_*"), key=lambda p: p.stat().st_mtime)
metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
classification = metrics["classification"]
baseline = float(classification["baseline"]["top1"])
replacement = float(classification["replacement"]["top1"])
train_dir = max(
    (p for p in root.glob("results/qkformer_cifar100_train_*") if (p / "checkpoint_manifest.json").exists()),
    key=lambda p: p.stat().st_mtime,
)
matches = re.findall(
    r"Best metric:\s*([0-9.]+)",
    (train_dir / "train_log.txt").read_text(encoding="utf-8", errors="replace"),
)
reported_best = max(map(float, matches))
if abs(baseline - replacement) > 1e-9:
    raise SystemExit(f"identity replacement mismatch: baseline={baseline}, replacement={replacement}")
if abs(baseline - reported_best) > 0.011:
    raise SystemExit(f"upstream evaluation mismatch: baseline={baseline}, reported_best={reported_best}")
print(f"[qk-lut-c100-t4-e3-protocol] identity_preflight=PASS baseline={baseline:.4f}")
PY

export QKFORMER_LUT_E3_ALPHA_INIT=0.025
export QKFORMER_LUT_E3_EPOCHS=2
bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh --gpu "$QKFORMER_LUT_GPU"
"$PYTHON_BIN" scripts/local/analyze_qk_lut_cifar100_t4.py
QKFORMER_LUT_PACKAGE_E3_COUNT=12 bash scripts/server/package_qkformer_lut_cifar100_t4.sh

echo "[qk-lut-c100-t4-e3-protocol] done=$(date -Is)"
