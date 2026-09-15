#!/bin/bash
set -euo pipefail
OUT=$(realpath "$1")
ROOT=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
export PATH=/opt/audio-runtime/ffmpeg/bin:/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_PRELOAD LD_LIBRARY_PATH PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
printf '%s.%s\n' "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$OUT/validation-step.txt"
cd "$OUT"
/opt/venv/bin/python "$ROOT/.worktrees/github-batch_tok/scripts/envs/image/validate_runtime.py" "$OUT"
cd "$OUT/installed-wheel-tests"
/opt/venv/bin/python -m pytest --import-mode=importlib -q \
    test/dataset/sampling test/dataset/test_checkpoint_restore.py test/test_indexing.py \
    test/shar/test_fork_runtime_compatibility.py test/shar/test_indexed_partition.py \
    --junitxml="$OUT/installed-wheel-tests.xml" > "$OUT/installed-wheel-tests.log" 2>&1
tail -5 "$OUT/installed-wheel-tests.log"
