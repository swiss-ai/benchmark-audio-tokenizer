#!/bin/bash
# Run inference for all datasets × 2 checkpoints × {transcribe, switch}
# 6 datasets × 2 checkpoints × 2 tasks = 24 runs

set -euo pipefail

MODEL_ROOT="/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-audio-S2-interleave-36B"
CHECKPOINTS=(
    "audio-weight-0-phase-transition"
    "audio-weight-1-phase-transition"
)
TASKS=("transcribe" "switch")

WAV_DIR="results/inference"
OUTPUT_DIR="results/inference_greedy"

# All datasets use --wav-dir under results/inference/{dataset_name}/
DATASETS=(
    "aishell1_test"
    "commonvoice_de"
    "commonvoice_fr"
    "eurospeech_uk"
    "fleurs_en_us"
    "spc_r_test"
)

for ckpt in "${CHECKPOINTS[@]}"; do
    MODEL_PATH="${MODEL_ROOT}/${ckpt}"
    for task in "${TASKS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            OUT="${OUTPUT_DIR}/${ds}/${ckpt}_${task}.json"
            if [ -f "${OUT}" ]; then
                echo "SKIP (exists): ${OUT}"
                continue
            fi
            mkdir -p "${OUTPUT_DIR}/${ds}"
            echo "========================================"
            echo "Dataset: ${ds} | Model: ${ckpt} | Task: ${task}"
            echo "========================================"
            python scripts/audio_inference.py \
                --model-path "${MODEL_PATH}" \
                --wav-dir "${WAV_DIR}/${ds}" \
                --dataset-name "${ds}" \
                --task "${task}" \
                --num-samples 10 \
                --output-file "${OUT}"
        done
    done
done

echo "All inference runs complete."
