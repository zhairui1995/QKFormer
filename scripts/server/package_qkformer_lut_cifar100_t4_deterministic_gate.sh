#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-qk_lutformer_cifar100_t4_deterministic_gate_artifacts.tar.gz}"
REPORT_PREFIX="${QKFORMER_LUT_GATE_OUTPUT_PREFIX:-results/qk_lutformer_cifar100_t4_deterministic_gate}"
CHECKPOINT_FILTER="${QKFORMER_LUT_CKPT:-}"
EPOCH_FILTER="${QKFORMER_LUT_E3_EPOCHS:-}"

mapfile -t FILES < <(python3 - "$ROOT" "$REPORT_PREFIX" "$CHECKPOINT_FILTER" "$EPOCH_FILTER" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
prefix = Path(sys.argv[2])
checkpoint_filter = sys.argv[3]
epoch_filter = sys.argv[4]
for path in (
    prefix.with_suffix(".json"),
    prefix.with_suffix(".md"),
    prefix.with_name(prefix.name + "_runs.csv"),
    prefix.with_name(prefix.name + "_summary.csv"),
):
    path = root / path
    if path.exists():
        print(path.relative_to(root))

dirs = []
for path in root.glob("results/qkformer_lut_e3_trainable_lut_*"):
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        continue
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    model = metrics.get("model", {})
    protocol = metrics.get("protocol", {})
    mode = metrics.get("adapter_summary", {}).get("mode")
    if model.get("family") != "cifar100" or str(model.get("time_step")) != "4":
        continue
    if int(protocol.get("version", 0)) < 5:
        continue
    if mode not in {"global_plus_address_lut", "global_plus_shuffled_address_lut", "global_mean"}:
        continue
    run_checkpoint = str(model.get("checkpoint", {}).get("path") or "")
    if checkpoint_filter and Path(run_checkpoint).resolve() != Path(checkpoint_filter).resolve():
        continue
    if epoch_filter and int(metrics.get("train_config", {}).get("epochs", -1)) != int(epoch_filter):
        continue
    if not metrics.get("gate_calibration", {}).get("per_sample"):
        continue
    dirs.append(path)

latest = {}
for path in dirs:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    key = (metrics.get("adapter_summary", {}).get("mode"), metrics.get("experiment", {}).get("seed"))
    if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
        latest[key] = path

for path in sorted(latest.values(), key=lambda p: str(p)):
    for name in ("metrics.json", "train_log.txt", "per_sample_predictions.csv", "gate_calibration_predictions.csv"):
        file_path = path / name
        if file_path.exists():
            print(file_path.relative_to(root))
PY
)

if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo "[qk-lut-c100-t4-gate-package] no files to package" >&2
  exit 1
fi

tar -czf "$OUT" "${FILES[@]}"
echo "[qk-lut-c100-t4-gate-package] wrote=$ROOT/$OUT"
