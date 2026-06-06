#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-qk_lutformer_t1_e3_seed_sweep_artifacts.tar.gz}"
NUM_E3="${QKFORMER_LUT_PACKAGE_E3_COUNT:-9}"

T1_TRAIN_DIR="$(python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.glob("results/qkformer_cifar10_train_*"):
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.exists():
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("time_step")) == "1":
        candidates.append(path)
if not candidates:
    raise SystemExit("no T=1 training result found")
print(max(candidates, key=lambda path: (path.stat().st_mtime, str(path))).relative_to(root))
PY
)"

T1_E0_DIR="$(python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.glob("results/qkformer_lut_e0_diag_*"):
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        continue
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if str(metrics.get("model", {}).get("time_step")) == "1":
        candidates.append(path)
if candidates:
    print(max(candidates, key=lambda path: (path.stat().st_mtime, str(path))).relative_to(root))
PY
)"

T1_E3_DIRS=()
while IFS= read -r dir; do
  T1_E3_DIRS+=("$dir")
done < <(python3 - "$ROOT" "$NUM_E3" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
limit = int(sys.argv[2])
candidates = []
for path in root.glob("results/qkformer_lut_e3_trainable_lut_*"):
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        continue
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if str(metrics.get("model", {}).get("time_step")) == "1":
        candidates.append(path)
for path in sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)[:limit]:
    print(path.relative_to(root))
PY
)

FILES=(
  "$T1_TRAIN_DIR/train_log.txt"
  "$T1_TRAIN_DIR/checkpoint_manifest.json"
)

SUMMARY_CSV="$(find "$T1_TRAIN_DIR" -name summary.csv | head -1)"
if [[ -n "$SUMMARY_CSV" ]]; then
  FILES+=("$SUMMARY_CSV")
fi

if [[ -n "$T1_E0_DIR" ]]; then
  FILES+=("$T1_E0_DIR/metrics.json" "$T1_E0_DIR/train_log.txt")
fi

for dir in "${T1_E3_DIRS[@]}"; do
  FILES+=("$dir/metrics.json" "$dir/train_log.txt")
done

tar -czf "$OUT" "${FILES[@]}"
if [[ "$OUT" = /* ]]; then
  echo "[qk-lut-package] wrote=$OUT"
else
  echo "[qk-lut-package] wrote=$ROOT/$OUT"
fi
