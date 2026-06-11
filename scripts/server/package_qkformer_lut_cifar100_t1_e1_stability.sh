#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-qk_lutformer_cifar100_t1_e1_stability_artifacts.tar.gz}"

mapfile -t FILES < <(python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
by_seed = {}
for path in root.glob("results/qkformer_lut_e1_recon_*"):
    metrics = path / "metrics.json"
    if not metrics.exists():
        continue
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    if payload.get("model", {}).get("family") != "cifar100":
        continue
    calibration = payload.get("data", {}).get("calibration", {})
    evaluation = payload.get("data", {}).get("evaluation", {})
    if not calibration.get("shuffle") or calibration.get("num_batches") != 128:
        continue
    if evaluation.get("num_batches", 0) < 300:
        continue
    seed = str(payload.get("experiment", {}).get("seed"))
    old = by_seed.get(seed)
    if old is None or path.stat().st_mtime > old.stat().st_mtime:
        by_seed[seed] = path

required = {"42", "43", "44"}
if set(by_seed) != required:
    raise SystemExit(f"expected seeds {sorted(required)}, found {sorted(by_seed)}")

for seed in sorted(by_seed):
    path = by_seed[seed]
    for name in ("metrics.json", "train_log.txt", "module_reconstruction.csv"):
        artifact = path / name
        if artifact.exists():
            print(artifact.relative_to(root))
PY
)

tar -czf "$OUT" "${FILES[@]}"
echo "[qk-lut-c100-e1-stability-package] wrote=$ROOT/$OUT"
