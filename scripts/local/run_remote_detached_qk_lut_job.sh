#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REMOTE="lbz@192.168.70.60"
REMOTE_ROOT="~/mac_agent/sdr-lutattn-qkformer-lut"
CONDA_ENV="sdr"
GIT_REMOTE="origin"
BRANCH="$(git branch --show-current)"
RUN_KIND="small"
GPU_DEFAULT="2"
JOB_NAME="qk_lut_job_$(date +%Y%m%d_%H%M%S)"
SERVER_SCRIPT=""
PACKAGE_SCRIPT=""
REMOTE_ENV=()

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/local/run_remote_detached_qk_lut_job.sh --server-script PATH [options]

Options:
  --remote USER@HOST       Default: lbz@192.168.70.60
  --remote-root PATH       Default: ~/mac_agent/sdr-lutattn-qkformer-lut
  --conda-env NAME         Default: sdr
  --git-remote NAME        Server-side Git remote. Default: origin
  --branch NAME            Default: current local branch
  --kind small|train       small uses --gpu; train auto-picks an idle GPU
  --gpu N                  Default GPU/fallback GPU. Default: 2
  --job-name NAME          Remote log/PID stem
  --env KEY=VALUE          Extra environment variable, repeatable
  --server-script PATH     Remote server script to run
  --package-script PATH    Optional package script to run after server script
  -h, --help               Show this help

Examples:
  bash scripts/local/run_remote_detached_qk_lut_job.sh \
    --kind train \
    --gpu 2 \
    --job-name c100_t4_seed42 \
    --env QKFORMER_LUT_TIME_STEP=4 \
    --env QKFORMER_CIFAR100_TRAIN_SEED=42 \
    --server-script scripts/server/run_qkformer_cifar100_train.sh
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
    --job-name)
      JOB_NAME="$2"; shift 2 ;;
    --env)
      REMOTE_ENV+=("$2"); shift 2 ;;
    --server-script)
      SERVER_SCRIPT="$2"; shift 2 ;;
    --package-script)
      PACKAGE_SCRIPT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[remote-detached] unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ "$RUN_KIND" != "small" && "$RUN_KIND" != "train" ]]; then
  echo "[remote-detached] --kind must be small or train" >&2
  exit 2
fi
if [[ -z "$SERVER_SCRIPT" ]]; then
  echo "[remote-detached] --server-script is required" >&2
  exit 2
fi

env_exports=""
for item in "${REMOTE_ENV[@]}"; do
  if [[ "$item" != *=* ]]; then
    echo "[remote-detached] --env must be KEY=VALUE, got: $item" >&2
    exit 2
  fi
  key="${item%%=*}"
  value="${item#*=}"
  if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "[remote-detached] invalid env key: $key" >&2
    exit 2
  fi
  printf -v quoted_value '%q' "$value"
  env_exports+="export ${key}=${quoted_value}; "
done

printf -v quoted_server_script '%q' "$SERVER_SCRIPT"
printf -v quoted_package_script '%q' "$PACKAGE_SCRIPT"
printf -v quoted_conda_env '%q' "$CONDA_ENV"
printf -v quoted_git_remote '%q' "$GIT_REMOTE"
printf -v quoted_branch '%q' "$BRANCH"
printf -v quoted_job_name '%q' "$JOB_NAME"
printf -v quoted_gpu_default '%q' "$GPU_DEFAULT"
printf -v quoted_run_kind '%q' "$RUN_KIND"

REMOTE_CMD="$(cat <<EOF
set -euo pipefail
cd $REMOTE_ROOT
mkdir -p results/remote_jobs
git fetch $quoted_git_remote $quoted_branch
git checkout $quoted_branch
git pull --ff-only $quoted_git_remote $quoted_branch
if [[ $quoted_run_kind == train ]]; then
  GPU=\$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '\$2+0<1000 && \$3+0<20 {gsub(/ /,"",\$1); print \$1; exit}' || true)
  GPU=\${GPU:-$quoted_gpu_default}
else
  GPU=$quoted_gpu_default
fi
LOG="results/remote_jobs/${quoted_job_name}.log"
PID_FILE="results/remote_jobs/${quoted_job_name}.pid"
RUNNER="source ~/miniconda/etc/profile.d/conda.sh; conda activate $quoted_conda_env; cd $REMOTE_ROOT; ${env_exports}echo [remote-detached] selected_gpu=\\\$GPU kind=$RUN_KIND; bash $quoted_server_script --gpu \\\$GPU"
if [[ -n $quoted_package_script ]]; then
  RUNNER="\$RUNNER && bash $quoted_package_script"
fi
nohup bash -lc "\$RUNNER" > "\$LOG" 2>&1 < /dev/null &
PID=\$!
echo "\$PID" > "\$PID_FILE"
echo "[remote-detached] job_name=$JOB_NAME"
echo "[remote-detached] pid=\$PID"
echo "[remote-detached] log=$REMOTE_ROOT/\$LOG"
echo "[remote-detached] pid_file=$REMOTE_ROOT/\$PID_FILE"
EOF
)"

echo "[remote-detached] remote=$REMOTE"
echo "[remote-detached] branch=$BRANCH"
echo "[remote-detached] server_script=$SERVER_SCRIPT"
ssh -o BatchMode=yes "$REMOTE" "$REMOTE_CMD"
