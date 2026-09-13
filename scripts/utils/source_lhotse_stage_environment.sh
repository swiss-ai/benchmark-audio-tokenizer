#!/bin/bash
# Source on the host; container setup belongs in source_lhotse_runtime.sh.

lhotse_srun() (
    set -euo pipefail
    local pipeline_stage=$1
    shift
    local stage_environment
    case "${pipeline_stage}" in
        tokenize)
            stage_environment="${TOKENIZE_ENVIRONMENT:-${REPO_DIR}/scripts/envs/nemo_26_08_audio_tokenization.toml}"
            export LHOTSE_RUNTIME_MODE=baked
            # Keep host Python installations and libraries outside the image.
            unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_PRELOAD LD_LIBRARY_PATH
            unset PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK HF_HUB_ENABLE_HF_TRANSFER
            export INSTALL_TORCHCODEC=0 INSTALL_TORCHAUDIO=0 INSTALL_ONCE_PER_NODE=0
            ;;
        convert|materialize)
            stage_environment="${LEGACY_ENVIRONMENT:-${REPO_DIR}/scripts/envs/nemo_25_11_audio_legacy.toml}"
            export LHOTSE_RUNTIME_MODE=legacy
            ;;
        *)
            echo "ERROR: unsupported pipeline stage: ${pipeline_stage}" >&2
            return 2
            ;;
    esac
    if [ ! -f "${stage_environment}" ]; then
        echo "ERROR: missing stage environment: ${stage_environment}" >&2
        return 2
    fi

    # srun must receive exactly one environment. These may be inherited from
    # an enclosing EDF/Pyxis step; keep the allocation and resource variables.
    local inherited_name
    while IFS= read -r inherited_name; do
        case "${inherited_name}" in
            *SPANK*|SLURM_EDF_*|PYXIS_*|ENROOT_*|SRUN_CONTAINER*|SRUN_ENVIRONMENT*|SBATCH_CONTAINER*|SBATCH_ENVIRONMENT*|SLURM_CONTAINER*|OCI_ANNOTATION_*|SLURM_EXPORT_ENV)
                unset "${inherited_name}"
                ;;
        esac
    done < <(compgen -e)

    echo "Runtime stage=${pipeline_stage} mode=${LHOTSE_RUNTIME_MODE} environment=${stage_environment} repo=${REPO_DIR}"
    srun --environment="${stage_environment}" "$@"
)
