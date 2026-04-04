#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=korean-tok
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/zeroth_korean_tokenize_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/zeroth_korean_tokenize_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

REPO=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer

echo "=========================================="
echo "Zeroth Korean — Direct Transcribe Tokenization"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname -s)"
echo "GPUs: 4"
echo "=========================================="

mkdir -p ${REPO}/logs

NODES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
HEAD_NODE=${NODES[0]}
MASTER_PORT=29500

srun -N 1 --tasks-per-node=1 -u bash -c '
    set -e

    REPO='"${REPO}"'
    export INSTALL_TORCHAUDIO=1
    source "${REPO}/scripts/source_lhotse_runtime.sh"

    cd "${REPO}"

    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr='"${HEAD_NODE}"' \
        --master_port='"${MASTER_PORT}"' \
        -m audio_tokenization.tokenize \
        mode=audio_text \
        audio_text_format=interleaved \
        audio_text_task=transcribe \
        dataset=audio_text/zeroth_korean_interleaved \
        num_gpus=4
'

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "=========================================="
