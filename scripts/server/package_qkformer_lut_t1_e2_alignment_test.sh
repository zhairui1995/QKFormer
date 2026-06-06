#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-qk_lutformer_t1_e2_alignment_test_artifacts.tar.gz}"

ALIGN_DIR="$(python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.glob("results/qkformer_lut_e2_alignment_test_*"):
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        continue
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if str(metrics.get("model", {}).get("time_step")) == "1":
        candidates.append(path)
if not candidates:
    raise SystemExit("no T=1 E2 alignment result found")
print(max(candidates, key=lambda path: (path.stat().st_mtime, str(path))).relative_to(root))
PY
)"

FILES=("$ALIGN_DIR/metrics.json" "$ALIGN_DIR/train_log.txt")

tar -czf "$OUT" "${FILES[@]}"
if tar -tzf "$OUT" | grep -E '\.(pth|pth\.tar|pt|ckpt)$' >/dev/null; then
  echo "[qk-lut-package-e2-align] refusing archive with checkpoint-like file" >&2
  rm -f "$OUT"
  exit 1
fi

if [[ "$OUT" = /* ]]; then
  echo "[qk-lut-package-e2-align] wrote=$OUT"
else
  echo "[qk-lut-package-e2-align] wrote=$ROOT/$OUT"
fi
