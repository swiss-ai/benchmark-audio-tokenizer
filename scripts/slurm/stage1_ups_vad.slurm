#!/bin/bash
# Stage-1 UPS / People's Speech VAD pipeline launcher.
#
# Convert raw WDS tar shards -> partitioned SHAR:
#   sbatch --time=12:00:00 --array=0-3 --nodes=1 --ntasks=1 --cpus-per-task=288 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/stage1_ups_vad.slurm convert
#
# Tokenize SHAR -> Megatron audio tokens:
#   sbatch --time=12:00:00 --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 \
#     --cpus-per-task=72 \
#     /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/slurm/stage1_ups_vad.slurm tokenize
#
# UPS is audio-only. Convert may run as a Slurm array; each task writes a
# disjoint child SHAR root:
#   ups_all_lang_shar_v3/node_00
#   ups_all_lang_shar_v3/node_01
#   ...
# Tokenization consumes the parent root and expands node_* automatically.
# materialize is intentionally disabled in the dataset card.
#
# Add normal Hydra overrides after the stage when needed, for example:
#   stage1_ups_vad.slurm tokenize runtime.resume=true

#SBATCH --account=infra01
#SBATCH --job-name=ups-stage1
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/ups_stage1_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/ups_stage1_%A_%a.err
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

DATASET=${DATASET:-stage1_ups_all_lang_vad}
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
WDS_ROOT=${WDS_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/unsupervised_peoples_speech_commercial_wds}
SHAR_ROOT=${SHAR_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1_rebuild/ups_all_lang_shar_v3}
NUM_WORKERS_PER_CONVERT_TASK=${NUM_WORKERS_PER_CONVERT_TASK:-${SLURM_CPUS_PER_TASK:-288}}

case "${STAGE}" in
    convert|tokenize) ;;
    *)
        echo "Unsupported stage: ${STAGE}" >&2
        echo "Use: convert or tokenize" >&2
        exit 2
        ;;
esac

echo "=========================================="
echo "UPS Stage-1 VAD Pipeline"
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
        if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
            mapfile -t ALL_TARS < <(find "${WDS_ROOT}/audio" "${WDS_ROOT}/audio2" -name '*.tar' | sort)
            TOTAL=${#ALL_TARS[@]}
            RANK=${SLURM_ARRAY_TASK_ID}
            WORLD=${SLURM_ARRAY_TASK_COUNT:-4}

            if [ "${TOTAL}" -eq 0 ]; then
                echo "ERROR: No WDS tar shards found under ${WDS_ROOT}/{audio,audio2}" >&2
                exit 1
            fi

            ASSIGNED_TARS=()
            for i in "${!ALL_TARS[@]}"; do
                if [ $((i % WORLD)) -eq "${RANK}" ]; then
                    ASSIGNED_TARS+=("${ALL_TARS[$i]}")
                fi
            done

            COUNT=${#ASSIGNED_TARS[@]}
            if [ "${COUNT}" -le 0 ]; then
                echo "[rank ${RANK}] No tar shards assigned (TOTAL=${TOTAL}, WORLD=${WORLD})"
                exit 0
            fi

            NODE_TAG=$(printf "%02d" "${RANK}")
            NODE_SHAR_DIR="${SHAR_ROOT}/node_${NODE_TAG}"
            TAR_LIST=$(IFS=,; echo "${ASSIGNED_TARS[*]}")

            echo "[rank ${RANK}/${WORLD}] assigned ${COUNT}/${TOTAL} WDS tar shards"
            echo "[rank ${RANK}/${WORLD}] output=${NODE_SHAR_DIR}"
            echo "[rank ${RANK}/${WORLD}] backend=${LHOTSE_AUDIO_BACKEND:-default}, resampling=soxr"

            HYDRA_OVERRIDES+=(
                "dataset.conversion.enabled=true"
                "dataset.conversion.num_workers=${NUM_WORKERS_PER_CONVERT_TASK}"
                "dataset.source.shards=[${TAR_LIST}]"
                "dataset.outputs.shar_dir=${NODE_SHAR_DIR}"
            )
        else
            HYDRA_OVERRIDES+=(
                "dataset.conversion.enabled=true"
                "dataset.conversion.num_workers=${NUM_WORKERS_PER_CONVERT_TASK}"
                "dataset.outputs.shar_dir=${SHAR_ROOT}"
            )
        fi
        run_stage convert 1
        ;;
    tokenize)
        run_stage tokenize "${TASKS}"
        ;;
esac
