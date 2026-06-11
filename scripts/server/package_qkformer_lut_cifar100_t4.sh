#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-qk_lutformer_cifar100_t4_artifacts.tar.gz}"
NUM_E3="${QKFORMER_LUT_PACKAGE_E3_COUNT:-12}"

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
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.exists():
        continue
    manifest = load_json(manifest_path)
    if str(manifest.get("time_step")) == "4":
        train_dirs.append(path)
if not train_dirs:
    raise SystemExit("no completed CIFAR-100 T=4 training result found")
train_dir = max(train_dirs, key=lambda p: (p.stat().st_mtime, str(p)))
for name in ("train_log.txt", "checkpoint_manifest.json"):
    path = train_dir / name
    if path.exists():
        print(path.relative_to(root))
summaries = list(train_dir.rglob("summary.csv"))
if summaries:
    print(summaries[0].relative_to(root))

def checkpoint_matches(payload):
    model = payload.get("model", {})
    if model.get("family") != "cifar100" or str(model.get("time_step")) != "4":
        return False
    ckpt = model.get("checkpoint", {})
    return str(ckpt.get("path") or "").startswith(str(train_dir))

for prefix, limit_count in (
    ("qkformer_lut_e0_diag_*", 1),
    ("qkformer_lut_e1_recon_*", 3),
    ("qkformer_lut_e3_trainable_lut_*", limit),
):
    dirs = []
    for path in root.glob(f"results/{prefix}"):
        metrics = path / "metrics.json"
        if not metrics.exists():
            continue
        if checkpoint_matches(load_json(metrics)):
            dirs.append(path)
    for path in sorted(dirs, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)[:limit_count]:
        for name in ("metrics.json", "train_log.txt", "module_reconstruction.csv"):
            file_path = path / name
            if file_path.exists():
                print(file_path.relative_to(root))
PY
)

if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo "[qk-lut-c100-t4-package] no files to package" >&2
  exit 1
fi

tar -czf "$OUT" "${FILES[@]}"
echo "[qk-lut-c100-t4-package] wrote=$ROOT/$OUT"
