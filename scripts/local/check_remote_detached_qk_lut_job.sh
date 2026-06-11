#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REMOTE="lbz@192.168.70.60"
REMOTE_ROOT="~/mac_agent/sdr-lutattn-qkformer-lut"
JOB_NAME=""
TAIL_LINES="80"
ARTIFACT=""
RESULTS_DIR="$ROOT/results"
EXTRACT=0
ANALYZE=0
KEEP_ARCHIVE=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/local/check_remote_detached_qk_lut_job.sh --job-name NAME [options]

Options:
  --remote USER@HOST       Default: lbz@192.168.70.60
  --remote-root PATH       Default: ~/mac_agent/sdr-lutattn-qkformer-lut
  --job-name NAME          Remote job stem used at launch
  --tail N                 Log lines to print. Default: 80
  --artifact NAME          Optional remote artifact to download from remote root
  --results-dir PATH       Local download directory. Default: repo results/
  --extract                Extract downloaded tarball into repo root
  --analyze                Run local qk_lut analyzers when available
  --keep-archive           Do not delete tarball after extraction
  -h, --help               Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="$2"; shift 2 ;;
    --remote-root)
      REMOTE_ROOT="$2"; shift 2 ;;
    --job-name)
      JOB_NAME="$2"; shift 2 ;;
    --tail)
      TAIL_LINES="$2"; shift 2 ;;
    --artifact)
      ARTIFACT="$2"; shift 2 ;;
    --results-dir)
      RESULTS_DIR="$2"; shift 2 ;;
    --extract)
      EXTRACT=1; shift ;;
    --analyze)
      ANALYZE=1; shift ;;
    --keep-archive)
      KEEP_ARCHIVE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[remote-detached-check] unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$JOB_NAME" ]]; then
  echo "[remote-detached-check] --job-name is required" >&2
  exit 2
fi

printf -v quoted_job_name '%q' "$JOB_NAME"
printf -v quoted_tail_lines '%q' "$TAIL_LINES"

REMOTE_CMD="$(cat <<EOF
set -euo pipefail
cd $REMOTE_ROOT
PID_FILE="results/remote_jobs/${quoted_job_name}.pid"
LOG="results/remote_jobs/${quoted_job_name}.log"
echo "[remote-detached-check] job_name=$JOB_NAME"
if [[ -f "\$PID_FILE" ]]; then
  PID=\$(cat "\$PID_FILE")
  echo "[remote-detached-check] pid=\$PID"
  if kill -0 "\$PID" >/dev/null 2>&1; then
    echo "[remote-detached-check] status=RUNNING"
  else
    echo "[remote-detached-check] status=NOT_RUNNING"
  fi
else
  echo "[remote-detached-check] status=NO_PID_FILE"
fi
if [[ -f "\$LOG" ]]; then
  echo "[remote-detached-check] log=$REMOTE_ROOT/\$LOG"
  echo "[remote-detached-check] --- tail ---"
  tail -n $quoted_tail_lines "\$LOG"
else
  echo "[remote-detached-check] log_missing=$REMOTE_ROOT/\$LOG"
fi
EOF
)"

ssh -o BatchMode=yes "$REMOTE" "$REMOTE_CMD"

if [[ -n "$ARTIFACT" ]]; then
  mkdir -p "$RESULTS_DIR"
  LOCAL_ARTIFACT="$RESULTS_DIR/$(basename "$ARTIFACT")"
  scp "${REMOTE}:${REMOTE_ROOT}/${ARTIFACT}" "$LOCAL_ARTIFACT"
  echo "[remote-detached-check] downloaded=$LOCAL_ARTIFACT"
  if [[ "$EXTRACT" -eq 1 ]]; then
    tar -xzf "$LOCAL_ARTIFACT" -C "$ROOT"
    echo "[remote-detached-check] extracted_to=$ROOT"
    if [[ "$KEEP_ARCHIVE" -eq 0 ]]; then
      rm -f "$LOCAL_ARTIFACT"
      echo "[remote-detached-check] removed_archive=$LOCAL_ARTIFACT"
    fi
  fi
fi

if [[ "$ANALYZE" -eq 1 ]]; then
  if [[ -x scripts/local/analyze_qk_lut_e6_budgeted_backoff.py || -f scripts/local/analyze_qk_lut_e6_budgeted_backoff.py ]]; then
    python3 scripts/local/analyze_qk_lut_e6_budgeted_backoff.py || true
  fi
  if [[ -x scripts/local/analyze_qk_lut_cifar100_t4.py || -f scripts/local/analyze_qk_lut_cifar100_t4.py ]]; then
    python3 scripts/local/analyze_qk_lut_cifar100_t4.py || true
  fi
  python3 scripts/local/analyze_qk_lut_results.py --root "$ROOT" --brief || true
fi
