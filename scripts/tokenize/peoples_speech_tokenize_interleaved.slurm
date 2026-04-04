#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ps-tok
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/ps_tokenize_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/ps_tokenize_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --exclusive

set -euo pipefail

REPO=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer

echo "=========================================="
echo "People's Speech Interleaved Tokenization"
echo "  Node: $(hostname -s), GPUs: 4"
echo "=========================================="

mkdir -p ${REPO}/logs

HEAD_NODE=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -1)

source "${REPO}/scripts/source_lhotse_runtime.sh"

cd "${REPO}"

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${HEAD_NODE}" \
    --master_port=29500 \
    -m audio_tokenization.tokenize \
    mode=audio_text \
    audio_text_format=interleaved \
    dataset=audio_text/peoples_speech_train_interleaved \
    num_gpus=4

echo "Tokenization completed!"
