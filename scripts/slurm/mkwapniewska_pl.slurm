#!/bin/bash
# Generic launcher for mkwapniewska Polish ASR pipeline datasets.
#
# Convert parquet -> SHAR:
#   sbatch --time=02:00:00 --nodes=1 --ntasks=1 --cpus-per-task=288 \
#     scripts/slurm/mkwapniewska_pl.slurm pl_granary_yodas_spk1636 convert
#
# Tokenize SHAR -> per-rank chunks:
#   sbatch --time=04:00:00 --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 \
#     --cpus-per-task=72 \
#     scripts/slurm/mkwapniewska_pl.slurm pl_granary_yodas_spk1636 tokenize \
#     outputs.tokenized_dir=/capstor/scratch/cscs/<USER>/tokenized

#SBATCH --account=infra01
#SBATCH --job-name=mk-pl
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/mk_pl_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/mk_pl_%j.err
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

SUBSET=${1:-}
STAGE=${2:-}
if [ -z "${SUBSET}" ] || [ -z "${STAGE}" ]; then
    echo "Usage: sbatch [resources] $0 <subset> <convert|tokenize> [hydra overrides...]" >&2
    echo "  subset: e.g. pl_granary_yodas_spk1636, pl_cv_spk1636, ..." >&2
    exit 2
fi
shift 2
HYDRA_OVERRIDES=("$@")
DATASET="mkwapniewska/${SUBSET}"

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

TASKS=${SLURM_NTASKS:-${SLURM_NPROCS:-1}}

case "${STAGE}" in
    convert|tokenize) ;;
    *)
        echo "Unsupported stage: ${STAGE}" >&2
        exit 2
        ;;
esac

echo "=========================================="
echo "mkwapniewska Polish ASR pipeline"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Dataset: ${DATASET}"
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

    echo "Running stage=${pipeline_stage} with srun --ntasks=${ntasks}"

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
esac
