#!/bin/bash
# Source from Slurm/bash jobs that need the local Lhotse runtime.

set -euo pipefail

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "ERROR: source ${BASH_SOURCE[0]} instead of executing it" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${REPO_DIR:=${DEFAULT_REPO_DIR}}"
: "${LHOTSE_DIR:=/iopsstor/scratch/cscs/xyixuan/dev/lhotse}"
: "${WHEELHOUSE_AARCH64:=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64}"
: "${FFMPEG_ROOT:=${WHEELHOUSE_AARCH64}/ffmpeg-7.1.1-full-aarch64}"
: "${TORCHCODEC_WHEELSET:=nemo_25_11}"
: "${TORCHCODEC_WHL:=${WHEELHOUSE_AARCH64}/${TORCHCODEC_WHEELSET}/torchcodec-0.9.0-cp312-cp312-linux_aarch64.whl}"
: "${TORCHAUDIO_SPEC:=git+https://github.com/pytorch/audio.git@release/2.9}"
: "${INSTALL_TORCHCODEC:=1}"
: "${INSTALL_TORCHAUDIO:=0}"
: "${INSTALL_ONCE_PER_NODE:=0}"
: "${PRINT_LHOTSE_RUNTIME:=1}"

export PYTHONPATH="${LHOTSE_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export PATH="/opt/venv/bin:${FFMPEG_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}"

# Pin BLAS/OpenMP to single-threaded for the convert stage. The prepare
# pipeline forks ~28+ workers per node; without this each forked worker
# spawns up to $(nproc) BLAS threads, causing 28×288 ≈ 8000 OS threads to
# thrash 288 cores. Probe 1985333 measured a 76× node-throughput regression
# without this pinning. Tokenize stages do GPU work and are left untouched.
if [ "${STAGE:-}" = "convert" ]; then
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
fi

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
