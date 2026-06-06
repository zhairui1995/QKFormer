#!/usr/bin/env bash

if [[ -n "${ROOT:-}" ]]; then
  case ":${PYTHONPATH:-}:" in
    *":$ROOT:"*) ;;
    *) export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
fi

qk_lut_select_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "$PYTHON"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    return 1
  fi
}

qk_lut_parse_gpu_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu)
        if [[ $# -lt 2 ]]; then
          echo "[qk-lut] --gpu requires an id, e.g. --gpu 2"
          return 2
        fi
        export QKFORMER_LUT_GPU="$2"
        shift 2
        ;;
      --gpu=*)
        export QKFORMER_LUT_GPU="${1#--gpu=}"
        shift
        ;;
      *)
        echo "[qk-lut] unknown argument: $1"
        return 2
        ;;
    esac
  done
}

qk_lut_configure_gpu() {
  if [[ -n "${QKFORMER_LUT_GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$QKFORMER_LUT_GPU"
  fi
}

qk_lut_log_gpu() {
  local python_bin="$1"
  local prefix="$2"
  echo "[$prefix] requested_gpu=${QKFORMER_LUT_GPU:-unset}"
  echo "[$prefix] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  "$python_bin" - "$prefix" <<'PY'
import sys

prefix = sys.argv[1]
try:
    import torch
except Exception as exc:
    print(f"[{prefix}] torch_gpu_probe_failed={exc}")
    raise SystemExit(0)

print(f"[{prefix}] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    current = torch.cuda.current_device()
    print(f"[{prefix}] torch_current_device={current}")
    print(f"[{prefix}] torch_current_device_name={torch.cuda.get_device_name(current)}")
    print(f"[{prefix}] torch_visible_device_count={torch.cuda.device_count()}")
PY
}
