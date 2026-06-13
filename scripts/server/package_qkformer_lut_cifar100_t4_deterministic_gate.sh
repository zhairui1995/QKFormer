#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-qk_lutformer_cifar100_t4_deterministic_gate_artifacts.tar.gz}"

mapfile -t FILES < <(python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in (
    "qk_lutformer_cifar100_t4_deterministic_gate.json",
    "qk_lutformer_cifar100_t4_deterministic_gate.md",
    "qk_lutformer_cifar100_t4_deterministic_gate_runs.csv",
    "qk_lutformer_cifar100_t4_deterministic_gate_summary.csv",
):
    path = root / "results" / name
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
