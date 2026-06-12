#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-qk_lutformer_cifar100_t4_selective_oracle_artifacts.tar.gz}"
NUM_E3="${QKFORMER_LUT_PACKAGE_E3_COUNT:-9}"

mapfile -t FILES < <(python3 - "$ROOT" "$NUM_E3" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
limit = int(sys.argv[2])

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

train_dirs = []
for path in root.glob("results/qkformer_cifar100_train_*"):
    manifest = load_json(path / "checkpoint_manifest.json")
    if str(manifest.get("time_step")) == "4":
        train_dirs.append(path)
if not train_dirs:
    raise SystemExit("no completed CIFAR-100 T=4 training result found")
train_dir = max(train_dirs, key=lambda p: (p.stat().st_mtime, str(p)))

for name in (
    "qk_lutformer_cifar100_t4_selective_oracle.json",
    "qk_lutformer_cifar100_t4_selective_oracle.md",
    "qk_lutformer_cifar100_t4_selective_oracle_runs.csv",
    "qk_lutformer_cifar100_t4_selective_oracle_summary.csv",
):
    path = root / "results" / name
    if path.exists():
        print(path.relative_to(root))

def matches(metrics):
    model = metrics.get("model", {})
    if model.get("family") != "cifar100" or str(model.get("time_step")) != "4":
        return False
    ckpt = model.get("checkpoint", {})
    if train_dir.name not in Path(str(ckpt.get("path") or "")).parts:
        return False
    return bool(metrics.get("classification", {}).get("per_sample"))

dirs = []
for path in root.glob("results/qkformer_lut_e3_trainable_lut_*"):
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        continue
    if matches(load_json(metrics_path)):
        dirs.append(path)

for path in sorted(dirs, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)[:limit]:
    for name in ("metrics.json", "train_log.txt", "per_sample_predictions.csv"):
        file_path = path / name
        if file_path.exists():
            print(file_path.relative_to(root))
PY
)

if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo "[qk-lut-c100-t4-oracle-package] no files to package" >&2
  exit 1
fi

tar -czf "$OUT" "${FILES[@]}"
echo "[qk-lut-c100-t4-oracle-package] wrote=$ROOT/$OUT"
