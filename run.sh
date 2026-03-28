#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=preprocess_libri
#SBATCH --environment=nemo
#SBATCH --output=/users/arsaikia/benchmark-audio-tokenizer-updated/librilogs/preprocess_%j.out
#SBATCH --error=/users/arsaikia/benchmark-audio-tokenizer-updated/librilogs/preprocess_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL

set -e

# --- API Key an OUTPUT --
WANDB_API_KEY="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"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/librilight/cutsets}"


REPO_DIR="${REPO_DIR:-/users/arsaikia/benchmark-audio-tokenizer-updated}"
GPUS_PER_NODE=4

echo "=========================================="
echo "Preprocess Pipeline"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Nodes:         ${SLURM_JOB_NUM_NODES}"
echo "Node list:     ${SLURM_JOB_NODELIST}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "OUTPUT_DIR:    ${OUTPUT_DIR}"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}/checkpoints"
mkdir -p "${REPO_DIR}/logs"

NODES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
HEAD_NODE=${NODES[0]}
MASTER_PORT=29500
TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))


srun -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=1 -u bash -c '
    set -e

    export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "compat" | paste -sd ":" -)
    if [ -d /usr/local/cuda/compat ]; then
        mv /usr/local/cuda/compat /usr/local/cuda/compat_disabled || true
    fi

    source '"${REPO_DIR}"'/nemo_venv/bin/activate
    export PYTHONPATH='"${REPO_DIR}"'/nemo_venv/lib/python3.12/site-packages:'"${REPO_DIR}"':$PYTHONPATH
    
    cd '"${REPO_DIR}"'
    export WANDB_API_KEY='"${WANDB_API_KEY}"'
    
    python -m torch.distributed.run \
        --nproc_per_node='"${GPUS_PER_NODE}"' \
        --nnodes='"${SLURM_JOB_NUM_NODES}"' \
        --node_rank=${SLURM_PROCID} \
        --master_addr='"${HEAD_NODE}"' \
        --master_port='"${MASTER_PORT}"' \
        -m preprocess.preprocess \
        output_dir='"${OUTPUT_DIR}"'/checkpoints \
        output_dir_data='"${OUTPUT_DIR}"' \
        resume=true \
        min_sample_rate=8000 \
        max_batch_duration=1000.0 \
        num_workers=4
'

echo ""
echo "=========================================="
echo "Preprocessing completed!"
echo "=========================================="
