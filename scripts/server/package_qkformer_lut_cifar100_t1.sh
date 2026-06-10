#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-qk_lutformer_cifar100_t1_artifacts.tar.gz}"
NUM_E3="${QKFORMER_LUT_PACKAGE_E3_COUNT:-12}"

mapfile -t FILES < <(python3 - "$ROOT" "$NUM_E3" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
limit = int(sys.argv[2])

train_dirs = [p for p in root.glob("results/qkformer_cifar100_train_*") if (p / "checkpoint_manifest.json").exists()]
if not train_dirs:
    raise SystemExit("no completed CIFAR-100 training result found")
train_dir = max(train_dirs, key=lambda p: (p.stat().st_mtime, str(p)))
for name in ("train_log.txt", "checkpoint_manifest.json"):
    path = train_dir / name
    if path.exists():
        print(path.relative_to(root))
summaries = list(train_dir.rglob("summary.csv"))
if summaries:
    print(summaries[0].relative_to(root))

e0_dirs = []
for path in root.glob("results/qkformer_lut_e0_diag_*"):
    metrics = path / "metrics.json"
    if not metrics.exists():
        continue
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    if payload.get("model", {}).get("family") == "cifar100":
        e0_dirs.append(path)
if e0_dirs:
    path = max(e0_dirs, key=lambda p: (p.stat().st_mtime, str(p)))
    print((path / "metrics.json").relative_to(root))
    print((path / "train_log.txt").relative_to(root))

e3_dirs = []
for path in root.glob("results/qkformer_lut_e3_trainable_lut_*"):
    metrics = path / "metrics.json"
    if not metrics.exists():
        continue
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    if payload.get("model", {}).get("family") == "cifar100":
        e3_dirs.append(path)
for path in sorted(e3_dirs, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)[:limit]:
    print((path / "metrics.json").relative_to(root))
    print((path / "train_log.txt").relative_to(root))
PY
)

tar -czf "$OUT" "${FILES[@]}"
echo "[qk-lut-c100-package] wrote=$ROOT/$OUT"
