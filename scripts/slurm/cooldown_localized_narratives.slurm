#!/bin/bash
# Cooldown Localized Narratives (Qwen-Audio retranscription) pipeline launcher.
#
# Convert processed parquet -> SHAR (audio_text_direct, transcribe task):
#   sbatch --time=06:00:00 --nodes=1 --ntasks=1 --cpus-per-task=288 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/cooldown_localized_narratives.slurm convert
#
# Tokenize SHAR -> Megatron audio tokens:
#   sbatch --time=12:00:00 --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 \
#     --cpus-per-task=72 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/cooldown_localized_narratives.slurm tokenize
#
# Source: /capstor/.../processed/localized_narratives_qwen/result_qwen_*.parquet
# 868,962 clips, ~5,220 hours (open_images dominates), 48 kHz mono, mean clip 21.6s.
# Audio is 16-bit OGG/Vorbis bytes; supervision text = caption_asr column.

#SBATCH --account=infra01
#SBATCH --job-name=ln-cooldown
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/cooldown_ln_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/cooldown_ln_%j.err
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

DATASET=cooldown/localized_narratives
STAGE=${1:-}
if [ -z "${STAGE}" ]; then
    echo "Usage: sbatch [resources] $0 <convert|tokenize> [hydra overrides...]" >&2
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
    convert|tokenize) ;;
    *)
        echo "Unsupported stage: ${STAGE}" >&2
        echo "Use: convert or tokenize" >&2
        exit 2
        ;;
esac

echo "=========================================="
echo "Localized Narratives (cooldown) Pipeline"
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

    # The dataset card pins conversion.enabled=false to gate convert behind an
    # explicit override; the launcher flips it on for the convert stage.
    local stage_override=()
    if [ "${pipeline_stage}" = "convert" ]; then
        stage_override+=("dataset.conversion.enabled=true")
    fi

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
' bash "${stage_override[@]}" "${HYDRA_OVERRIDES[@]}"
}

case "${STAGE}" in
    convert)
        run_stage convert 1
        ;;
    tokenize)
        run_stage tokenize "${TASKS}"
        ;;
esac
