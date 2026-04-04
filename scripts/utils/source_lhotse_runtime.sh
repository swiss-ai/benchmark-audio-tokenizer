#!/bin/bash
# Source from Slurm/bash jobs that need the local Lhotse runtime.

set -euo pipefail

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "ERROR: source ${BASH_SOURCE[0]} instead of executing it" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${REPO_DIR:=${DEFAULT_REPO_DIR}}"
: "${LHOTSE_DIR:=/iopsstor/scratch/cscs/xyixuan/dev/lhotse}"
: "${FFMPEG_ROOT:=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64}"
: "${TORCHCODEC_WHL:=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/torchcodec-0.9.0-cp312-cp312-linux_aarch64.whl}"
: "${TORCHAUDIO_SPEC:=git+https://github.com/pytorch/audio.git@release/2.9}"
: "${INSTALL_TORCHCODEC:=1}"
: "${INSTALL_TORCHAUDIO:=0}"
: "${INSTALL_ONCE_PER_NODE:=0}"
: "${PRINT_LHOTSE_RUNTIME:=1}"

export PYTHONPATH="${LHOTSE_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${FFMPEG_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}"

lhotse_runtime_install() {
  if [ "${INSTALL_TORCHCODEC}" = "1" ]; then
    uv pip install --python /opt/venv/bin/python --no-deps --force-reinstall "${TORCHCODEC_WHL}"
  fi
  if [ "${INSTALL_TORCHAUDIO}" = "1" ]; then
    uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation "${TORCHAUDIO_SPEC}"
  fi
}

lhotse_runtime_install_once_per_node() {
  READY_FILE="${RUNTIME_READY_FILE:-/tmp/.lhotse_runtime_ready_${SLURM_JOB_ID:-$$}}"
  if [ "${SLURM_LOCALID:-0}" = "0" ]; then
    lhotse_runtime_install
    touch "${READY_FILE}"
  else
    while [ ! -f "${READY_FILE}" ]; do
      sleep 1
    done
  fi
}

lhotse_runtime_debug() {
  python -c "import lhotse, sys; print(f'lhotse={lhotse.__file__}'); print('sys.path[0:5]=\n' + '\n'.join('  ' + p for p in sys.path[:5]))"
}

SHOULD_INSTALL=0
if [ "${INSTALL_TORCHCODEC}" = "1" ] || [ "${INSTALL_TORCHAUDIO}" = "1" ]; then
  SHOULD_INSTALL=1
fi

if [ "${SHOULD_INSTALL}" = "1" ]; then
  if [ "${INSTALL_ONCE_PER_NODE}" = "1" ]; then
    lhotse_runtime_install_once_per_node
  else
    lhotse_runtime_install
  fi
fi

if [ "${PRINT_LHOTSE_RUNTIME}" = "1" ]; then
  lhotse_runtime_debug
fi
