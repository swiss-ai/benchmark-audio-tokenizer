#!/bin/bash
# Cooldown AMI IHM (Individual Headset Mics) pipeline launcher.
#
# Convert parquet -> SHAR (interleaved metadata):
#   sbatch --time=01:00:00 --nodes=1 --ntasks=1 --cpus-per-task=288 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/cooldown_ami.slurm convert
#
# Tokenize SHAR -> per-rank chunks:
#   sbatch --time=03:00:00 --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 \
#     --cpus-per-task=72 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/cooldown_ami.slurm tokenize
#
# Materialize interleaved sequences:
#   sbatch --time=02:00:00 --nodes=1 --ntasks=1 --cpus-per-task=288 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/cooldown_ami.slurm materialize
#
# Source: /capstor/.../raw/ami/ihm/train-*.parquet (42 shards)
# 108,502 turns, ~100 h, English meetings.
# Interleave: source_id = meeting_id, clip_start = begin_time (real seconds);
# clip_num derived from timestamp. SDM split intentionally excluded.

#SBATCH --account=infra01
#SBATCH --job-name=ami-cooldown
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/cooldown_ami_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/cooldown_ami_%j.err
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

DATASET=cooldown/ami
STAGE=${1:-}
if [ -z "${STAGE}" ]; then
    echo "Usage: sbatch [resources] $0 <convert|tokenize|materialize> [hydra overrides...]" >&2
    exit 2
fi
shift
HYDRA_OVERRIDES=("$@")

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

TASKS=${SLURM_NTASKS:-${SLURM_NPROCS:-1}}

case "${STAGE}" in
    convert|tokenize|materialize) ;;
    *)
        echo "Unsupported stage: ${STAGE}" >&2
        echo "Use: convert, tokenize, or materialize" >&2
        exit 2
        ;;
esac

echo "=========================================="
echo "AMI IHM (cooldown) Pipeline"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Stage: ${STAGE}"
echo "Tasks: ${TASKS}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Hydra overrides: ${HYDRA_OVERRIDES[*]:-(none)}"
echo "=========================================="

export REPO_DIR RUNTIME_ENV_SH DATASET STAGE
export INSTALL_TORCHAUDIO=${INSTALL_TORCHAUDIO:-1}
export INSTALL_TORCHCODEC=${INSTALL_TORCHCODEC:-0}
export INSTALL_ONCE_PER_NODE=${INSTALL_ONCE_PER_NODE:-1}
export PRINT_LHOTSE_RUNTIME=${PRINT_LHOTSE_RUNTIME:-0}
export RUNTIME_READY_FILE=${RUNTIME_READY_FILE:-/tmp/.lhotse_runtime_ready_${SLURM_JOB_ID:-manual}}
if [ "${STAGE}" = "convert" ]; then
    export LHOTSE_AUDIO_BACKEND=${LHOTSE_AUDIO_BACKEND:-LibsndfileBackend}
fi

run_stage() {
    local pipeline_stage=$1
    local ntasks=$2
    shift 2

    echo "=========================================="
    echo "Running stage=${pipeline_stage} with srun --ntasks=${ntasks} $*"
    echo "=========================================="

    STAGE="${pipeline_stage}" srun --kill-on-bad-exit=0 -u --ntasks="${ntasks}" "$@" bash -c '
set -euo pipefail

jemalloc=/opt/venv/lib/python3.12/site-packages/ray/core/libjemalloc.so
if [ -f "${jemalloc}" ]; then
    export LD_PRELOAD="${jemalloc}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

source "${RUNTIME_ENV_SH}"
cd "${REPO_DIR}"

python -m audio_tokenization run \
    dataset="${DATASET}" \
    stage="${STAGE}" \
    "$@"
' bash "${HYDRA_OVERRIDES[@]}"
}

case "${STAGE}" in
    convert)
        run_stage convert 1
        ;;
    tokenize)
        run_stage tokenize "${TASKS}"
        ;;
    materialize)
        run_stage materialize 1
        ;;
esac
