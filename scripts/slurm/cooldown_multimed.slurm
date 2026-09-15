#!/bin/bash
# Cooldown MultiMed (medical-domain ASR, 5 langs) pipeline launcher.
#
# Convert parquet -> SHAR:
#   sbatch --time=00:30:00 --nodes=1 --ntasks=1 --cpus-per-task=288 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/cooldown_multimed.slurm convert
#
# Tokenize SHAR -> per-rank chunks:
#   sbatch --time=02:00:00 --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 \
#     --cpus-per-task=72 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/cooldown_multimed.slurm tokenize
#
# Source: /capstor/.../processed/multimed/data/{de,en,fr,vi,zh}.parquet
# Built by multimodal-data/01-dataset-download/audio/multimed/repack_with_locale.py
# from raw /raw/multimed/{Chinese,English,French,German,Vietnamese}/train-*.parquet.
# 34,148 clips, ~98.4 h, 5 languages with `locale` column carried per-row.

#SBATCH --account=infra01
#SBATCH --job-name=mm-cooldown
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/cooldown_multimed_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/cooldown_multimed_%j.err
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

DATASET=cooldown/multimed
STAGE=${1:-}
if [ -z "${STAGE}" ]; then
    echo "Usage: sbatch [resources] $0 <convert|tokenize|materialize> [hydra overrides...]" >&2
    exit 2
fi
shift
HYDRA_OVERRIDES=("$@")

# sbatch spools this file, so BASH_SOURCE cannot identify the checkout there.
# Submit from the checkout, or export REPO_DIR explicitly before sbatch.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "${REPO_DIR}" ]; then
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [ ! -f "${REPO_DIR}/audio_tokenization/__main__.py" ] || \
   [ ! -f "${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh" ]; then
    echo "ERROR: REPO_DIR must name the pipeline checkout; got ${REPO_DIR}" >&2
    exit 2
fi
REPO_DIR="$(cd "${REPO_DIR}" && pwd)"
source "${REPO_DIR}/scripts/utils/source_lhotse_stage_environment.sh"
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
echo "MultiMed (cooldown) Pipeline"
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

    STAGE="${pipeline_stage}" lhotse_srun "${pipeline_stage}" --kill-on-bad-exit=0 -u --ntasks="${ntasks}" "$@" bash -c '
set -euo pipefail

jemalloc=/opt/venv/lib/python3.12/site-packages/ray/core/libjemalloc.so
if [ "${LHOTSE_RUNTIME_MODE:-legacy}" = "legacy" ] && [ -f "${jemalloc}" ]; then
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
