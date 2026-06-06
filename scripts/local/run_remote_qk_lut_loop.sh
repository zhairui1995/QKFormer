#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REMOTE="lbz@192.168.70.60"
REMOTE_ROOT="~/mac_agent/sdr-lutattn-qkformer-lut"
CONDA_ENV="sdr"
GIT_REMOTE="origin"
BRANCH="codex/qkformer-lut-hybrid"
RUN_KIND="small"
GPU_DEFAULT="2"
SERVER_SCRIPT="scripts/server/run_qkformer_lut_t1_e3_seed_sweep.sh"
PACKAGE_SCRIPT="scripts/server/package_qkformer_lut_t1_e3_seed_sweep.sh"
ARTIFACT="qk_lutformer_t1_e3_seed_sweep_artifacts.tar.gz"
RESULTS_DIR="$ROOT/results"
DOWNLOAD=1
EXTRACT=1
ANALYZE=1
KEEP_ARCHIVE=0
SKIP_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/local/run_remote_qk_lut_loop.sh [options]

Options:
  --remote USER@HOST            Default: lbz@192.168.70.60
  --remote-root PATH            Default: ~/mac_agent/sdr-lutattn-qkformer-lut
  --conda-env NAME              Default: sdr
  --git-remote NAME             Server-side Git remote. Default: origin
  --branch NAME                 Default: codex/qkformer-lut-hybrid
  --kind small|train            small uses --gpu directly; train auto-picks an idle GPU first
  --gpu N                       Default GPU for small jobs and fallback GPU for train jobs
  --server-script PATH          Remote script to run
  --package-script PATH         Remote package script to run after server script
  --artifact NAME               Artifact path/name produced on remote root
  --results-dir PATH            Local download directory; default: repo results/
  --no-download                 Run remote command only
  --no-extract                  Download but do not extract
  --no-analyze                  Skip local metrics summary
  --keep-archive                Keep downloaded tar.gz after extraction
  --skip-run                    Pull remote branch and package/download existing results only
  -h, --help                    Show this help

Examples:
  bash scripts/local/run_remote_qk_lut_loop.sh
  bash scripts/local/run_remote_qk_lut_loop.sh --kind train --server-script scripts/server/run_qkformer_cifar10_t1_train.sh --artifact qk_t1_train.tar.gz --no-download
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="$2"; shift 2 ;;
    --remote-root)
      REMOTE_ROOT="$2"; shift 2 ;;
    --conda-env)
      CONDA_ENV="$2"; shift 2 ;;
    --git-remote)
      GIT_REMOTE="$2"; shift 2 ;;
    --branch)
      BRANCH="$2"; shift 2 ;;
    --kind)
      RUN_KIND="$2"; shift 2 ;;
    --gpu)
      GPU_DEFAULT="$2"; shift 2 ;;
    --server-script)
      SERVER_SCRIPT="$2"; shift 2 ;;
    --package-script)
      PACKAGE_SCRIPT="$2"; shift 2 ;;
    --artifact)
      ARTIFACT="$2"; shift 2 ;;
    --results-dir)
      RESULTS_DIR="$2"; shift 2 ;;
    --no-download)
      DOWNLOAD=0; shift ;;
    --no-extract)
      EXTRACT=0; shift ;;
    --no-analyze)
      ANALYZE=0; shift ;;
    --keep-archive)
      KEEP_ARCHIVE=1; shift ;;
    --skip-run)
      SKIP_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[remote-loop] unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ "$RUN_KIND" != "small" && "$RUN_KIND" != "train" ]]; then
  echo "[remote-loop] --kind must be small or train" >&2
  exit 2
fi

if [[ -z "$SERVER_SCRIPT" && "$SKIP_RUN" -eq 0 ]]; then
  echo "[remote-loop] --server-script cannot be empty" >&2
  exit 2
fi

REMOTE_CMD="$(cat <<EOF
set -euo pipefail
cd $REMOTE_ROOT
git fetch $GIT_REMOTE $BRANCH
git checkout $BRANCH
git pull --ff-only $GIT_REMOTE $BRANCH
source ~/miniconda/etc/profile.d/conda.sh
conda activate $CONDA_ENV
if [[ "$RUN_KIND" == "train" ]]; then
  GPU=\$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '\$2+0<1000 && \$3+0<20 {gsub(/ /,"",\$1); print \$1; exit}' || true)
  GPU=\${GPU:-$GPU_DEFAULT}
else
  GPU="$GPU_DEFAULT"
fi
echo "[remote-loop] selected_gpu=\$GPU kind=$RUN_KIND"
if [[ "$SKIP_RUN" -eq 0 ]]; then
  bash "$SERVER_SCRIPT" --gpu "\$GPU"
else
  echo "[remote-loop] server run skipped"
fi
if [[ -n "$PACKAGE_SCRIPT" ]]; then
  bash "$PACKAGE_SCRIPT" "$ARTIFACT"
fi
EOF
)"

echo "[remote-loop] remote=$REMOTE"
echo "[remote-loop] branch=$BRANCH"
echo "[remote-loop] server_script=$SERVER_SCRIPT"
echo "[remote-loop] package_script=${PACKAGE_SCRIPT:-none}"
echo "[remote-loop] artifact=$ARTIFACT"

ssh -o BatchMode=yes "$REMOTE" "$REMOTE_CMD"

if [[ "$DOWNLOAD" -eq 0 ]]; then
  echo "[remote-loop] download skipped"
  exit 0
fi

mkdir -p "$RESULTS_DIR"
LOCAL_ARTIFACT="$RESULTS_DIR/$(basename "$ARTIFACT")"
scp "${REMOTE}:${REMOTE_ROOT}/${ARTIFACT}" "$LOCAL_ARTIFACT"
echo "[remote-loop] downloaded=$LOCAL_ARTIFACT"

if [[ "$EXTRACT" -eq 1 ]]; then
  tar -xzf "$LOCAL_ARTIFACT" -C "$ROOT"
  echo "[remote-loop] extracted_to=$ROOT"
  if [[ "$KEEP_ARCHIVE" -eq 0 ]]; then
    rm -f "$LOCAL_ARTIFACT"
    echo "[remote-loop] removed_archive=$LOCAL_ARTIFACT"
  fi
fi

if [[ "$ANALYZE" -eq 1 ]]; then
  python3 scripts/local/analyze_qk_lut_results.py --root "$ROOT"
fi
