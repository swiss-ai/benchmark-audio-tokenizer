#!/bin/bash
# Submit convert+tokenize as a dependent chain for one mkwapniewska Polish
# subset.
#
# Usage:
#   scripts/slurm/mkwapniewska_pl_chain.sh <subset>
#
# Example:
#   scripts/slurm/mkwapniewska_pl_chain.sh pl_granary_yodas_spk1636
#
# Walltime defaults are sized for the small subsets. For pl_granary_1kh_spk1636
# (850 h) pass --convert-time and --tokenize-time, e.g.:
#   CONVERT_TIME=04:00:00 TOKENIZE_TIME=08:00:00 \
#     scripts/slurm/mkwapniewska_pl_chain.sh pl_granary_1kh_spk1636

set -euo pipefail

SUBSET=${1:-}
if [ -z "${SUBSET}" ]; then
    echo "Usage: $0 <subset>" >&2
    echo "  subsets: pl_cv_spk1636, pl_mls_spk1636," >&2
    echo "           pl_granary_yodas_spk1636, pl_granary_ytc_spk1636," >&2
    echo "           pl_granary_1kh_spk1636" >&2
    exit 2
fi

REPO=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
LAUNCHER="${REPO}/scripts/slurm/mkwapniewska_pl.slurm"
CONVERT_TIME=${CONVERT_TIME:-02:00:00}
TOKENIZE_TIME=${TOKENIZE_TIME:-04:00:00}

# Pyxis refuses duplicate --environment when SPANK envvar is set by the
# interactive container — strip it so sbatch can re-set it.
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment 2>/dev/null || true

JOB_CONVERT=$(sbatch --parsable \
    --reservation=SD-69241-apertus-1-5-0 \
    --time="${CONVERT_TIME}" \
    --nodes=1 --ntasks=1 --cpus-per-task=288 \
    "${LAUNCHER}" "${SUBSET}" convert)
echo "convert  job: ${JOB_CONVERT}"

JOB_TOKENIZE=$(sbatch --parsable \
    --dependency=afterok:"${JOB_CONVERT}" --kill-on-invalid-dep=yes \
    --reservation=SD-69241-apertus-1-5-0 \
    --time="${TOKENIZE_TIME}" \
    --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=72 \
    "${LAUNCHER}" "${SUBSET}" tokenize)
echo "tokenize job: ${JOB_TOKENIZE}  (waits for convert)"
echo
echo "Logs under: ${REPO}/logs/mk_pl_{${JOB_CONVERT},${JOB_TOKENIZE}}.{out,err}"
