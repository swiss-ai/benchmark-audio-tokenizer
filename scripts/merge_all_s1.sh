#!/bin/bash
# Filter (remove sequences with <5 unique audio tokens) and merge per-rank
# chunks within each dataset in parallel.
# Each dataset gets its own filtered+merged .bin/.idx under merged_s1/.
set -euo pipefail

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}

INPUT_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized
OUTPUT_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/merged_s1

# Audio token offset — needs to match the omni-tokenizer used during tokenization
TOKENIZER_PATH=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok

mkdir -p ${OUTPUT_ROOT}

for dataset_dir in ${INPUT_ROOT}/*/; do
    dataset_name=$(basename ${dataset_dir})
    output_prefix=${OUTPUT_ROOT}/${dataset_name}
    echo "[$(date '+%F %T')] Merging ${dataset_name} ..."
    python -m audio_tokenization.utils.indexed_dataset.audio_filter_and_merge \
        --input-dirs ${dataset_dir} \
        --output-prefix ${output_prefix} \
        --min-unique-tokens 5 \
        --tokenizer-path ${TOKENIZER_PATH} \
        --recursive \
        --stats-json ${OUTPUT_ROOT}/${dataset_name}_stats.json &
done

echo "Waiting for all merges to finish..."
wait
echo "[$(date '+%F %T')] All done. Output: ${OUTPUT_ROOT}"
