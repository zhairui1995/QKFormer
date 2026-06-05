#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export QKFORMER_LUT_TIME_STEP=1
export QKFORMER_TRAIN_EXPERIMENT="${QKFORMER_TRAIN_EXPERIMENT:-qkformer_cifar10_t1}"

bash scripts/server/run_qkformer_cifar10_train.sh "$@"
