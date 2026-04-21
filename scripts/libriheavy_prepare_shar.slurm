#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=libriheavy-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/libriheavy_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/libriheavy_shar_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --exclusive

set -euo pipefail

REPO=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
PARQUET_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___mythicinfinity___libriheavy/default/large
SHAR_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/libriheavy_large

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD="${SLURM_ARRAY_TASK_COUNT}"
SHARD_ID=$(printf "shard-%05d-of-0020" ${RANK})

NODE_TAG="$(printf "%02d" "${RANK}")"
NODE_SHAR_DIR="${SHAR_ROOT}/node_${NODE_TAG}"

echo "=========================================="
echo "LibriHeavy Large → Shar (${SHARD_ID})"
echo "  Rank ${RANK}/${WORLD}, 32 workers"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname -s)"
echo "Input: ${PARQUET_ROOT}/${SHARD_ID}"
echo "Output: ${NODE_SHAR_DIR}"
echo "=========================================="

mkdir -p ${REPO}/logs "${NODE_SHAR_DIR}"

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:${REPO}:${PYTHONPATH:-}
export PATH=/users/xyixuan/.local/ffmpeg/bin:$PATH
export LD_LIBRARY_PATH=/users/xyixuan/.local/ffmpeg/lib:${LD_LIBRARY_PATH:-}

ulimit -n 65536
uv pip install --python /opt/venv/bin/python polars

python -m audio_tokenization.prepare.prepare_parquet_to_shar \
    --parquet-dir "${PARQUET_ROOT}/${SHARD_ID}" \
    --parquet-glob '*.parquet' \
    --shar-dir "${NODE_SHAR_DIR}" \
    --id-column id \
    --audio-column audio \
    --text-column text_original \
    --duration-column audio_duration \
    --custom-columns speaker_id librivox_book_id \
    --text-tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
    --target-sr 24000 \
    --shard-size 20000 \
    --num-workers 32

echo ""
echo "=========================================="
echo "Rank ${RANK} (${SHARD_ID}) completed!"
echo "=========================================="

# Last task aggregates stats from all nodes
if [ "${RANK}" -eq "$((WORLD - 1))" ]; then
    echo "Last rank — waiting for siblings then aggregating..."
    # Submit aggregate as dependent job (runs after all array tasks finish)
    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
        --account=infra01 --partition=normal --reservation=PA-2338-RL \
        --environment=nemo_25_11 \
        --time=00:10:00 --nodes=1 --cpus-per-task=1 \
        --job-name=libriheavy-agg \
        --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/libriheavy_aggregate_%j.out \
        --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/libriheavy_aggregate_%j.err \
        --wrap="export PYTHONPATH=${REPO}:/iopsstor/scratch/cscs/xyixuan/dev/lhotse:\${PYTHONPATH:-}; \
                python -m audio_tokenization.aggregate_prepare \
                    --shar-root ${SHAR_ROOT}"
    echo "Aggregate job submitted"
fi
