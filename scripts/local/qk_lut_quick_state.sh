#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

sed -n '1,220p' docs/QK_LUTFORMER_QUICK_STATE.md

if [[ -d "$ROOT/results" ]]; then
  echo
  python3 scripts/local/analyze_qk_lut_results.py --root "$ROOT" --brief
fi
