#!/bin/bash
# Generic SFT audio pipeline launcher.
#
# Convert embedded-audio parquet -> SHAR:
#   sbatch --reservation=SD-69241-apertus-1-5-0 --time=04:00:00 \
#     --nodes=1 --ntasks=1 --cpus-per-task=288 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/sft_audio.slurm \
#     sft/teleantifraud_matching convert
#
# Tokenize SHAR -> reusable audio-token cache:
#   sbatch --reservation=SD-69241-apertus-1-5-0 --time=04:00:00 \
#     --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=72 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/sft_audio.slurm \
#     sft/teleantifraud_matching tokenize
#
# Materialize SFT conversations from conversation parquets + audio-token cache:
#   sbatch --reservation=SD-69241-apertus-1-5-0 --time=02:00:00 \
#     --nodes=1 --ntasks=1 --cpus-per-task=288 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/sft_audio.slurm \
#     sft/teleantifraud_matching materialize
#
# The same launcher works for any SFT dataset card, for example:
#   .../scripts/slurm/sft_audio.slurm sft/marco_longspeech convert

#SBATCH --account=infra01
#SBATCH --job-name=sft-audio
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/sft_audio_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/sft_audio_%j.err
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

DATASET=${1:-}
STAGE=${2:-}
if [ -z "${DATASET}" ] || [ -z "${STAGE}" ]; then
    echo "Usage: sbatch [resources] $0 <dataset> <convert|tokenize|materialize> [hydra overrides...]" >&2
    exit 2
fi
shift 2
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
echo "SFT Audio Pipeline"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Dataset: ${DATASET}"
echo "Stage: ${STAGE}"
echo "Tasks: ${TASKS}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Hydra overrides: ${HYDRA_OVERRIDES[*]:-(none)}"
echo "=========================================="

export REPO_DIR RUNTIME_ENV_SH DATASET STAGE
export INSTALL_TORCHCODEC=${INSTALL_TORCHCODEC:-0}
export INSTALL_TORCHAUDIO=${INSTALL_TORCHAUDIO:-1}
export INSTALL_ONCE_PER_NODE=${INSTALL_ONCE_PER_NODE:-1}
export PRINT_LHOTSE_RUNTIME=${PRINT_LHOTSE_RUNTIME:-0}
export RUNTIME_READY_FILE=${RUNTIME_READY_FILE:-/tmp/.lhotse_runtime_ready_${SLURM_JOB_ID:-manual}}
if [ "${STAGE}" = "convert" ]; then
    export LHOTSE_AUDIO_BACKEND=${LHOTSE_AUDIO_BACKEND:-LibsndfileBackend}
fi

run_stage() {
    local pipeline_stage=$1
    local ntasks=$2

    echo "=========================================="
    echo "Running dataset=${DATASET} stage=${pipeline_stage} with srun --ntasks=${ntasks}"
    echo "=========================================="

    STAGE="${pipeline_stage}" srun --kill-on-bad-exit=0 -u --ntasks="${ntasks}" bash -c '
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
