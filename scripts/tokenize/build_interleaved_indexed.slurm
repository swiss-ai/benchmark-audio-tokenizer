#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=build-idx
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/logs/build_idx_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/logs/build_idx_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --array=0-5

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok
BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized/audio_text_interleaved
LOG_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/logs
mkdir -p "${LOG_DIR}"

DATASETS=(
    coral_v3_conversation_train_interleaved_dur0p1-200
    emilia_yodas_interleaved_dur2-200
    legco_speech_train_interleaved_dur0p1-200
    mls_8lang_interleaved_dur0-200
    peoples_speech_train_interleaved_dur0p1-200
    wenetspeech_interleaved_dur0-200
)

DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
PARQUET_DIR="${BASE}/${DS}/parquet"
OUTPUT_DIR="${BASE}/${DS}/indexed"

echo "=========================================="
echo "Build Interleaved Indexed: ${DS}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID} (array ${SLURM_ARRAY_TASK_ID})"
echo "Parquet dir: ${PARQUET_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "=========================================="

uv pip install --python /opt/venv/bin/python polars

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python -m audio_tokenization.utils.build_interleaved_indexed \
    --parquet-dir "${PARQUET_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --tokenizer-path "${TOKENIZER}" \
    --patterns ATAT TATA \
    --num-workers 32

echo "=========================================="
echo "Done: ${DS}"
echo "=========================================="
