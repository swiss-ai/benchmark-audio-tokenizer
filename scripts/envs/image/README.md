# NeMo 26.08 WavTokenizer pipeline image

The dataset launchers use one image for conversion, tokenization and
materialization. It contains our pinned Lhotse wheel, CUDA-enabled TorchAudio,
SoXR, Polars and FFmpeg. Jobs import installed libraries and do not compile or
install dependencies at startup. Preparation uses public Lhotse SHAR APIs;
the pipeline publishes completed partitions atomically.

## Lhotse source and version

Our fork is the Git submodule at `src/3rdparty/lhotse`:

```sh
git submodule update --init src/3rdparty/lhotse
git -C src/3rdparty/lhotse rev-parse HEAD
```

The pin is `89ed3be63756b0f1d87c5cfd0af1e9639527f557` from
[Alvorecer721/lhotse](https://github.com/Alvorecer721/lhotse/tree/89ed3be63756b0f1d87c5cfd0af1e9639527f557).
It includes the padded-duration sampler fix, upstream integration and the
StatelessSampler initialization correction. The package is still named
`lhotse`; its wheel version is `2.0.0a6.dev0+git.89ed3be6.clean`.

The gitlink is an exact commit. Updating a branch in the fork does not update
this pipeline automatically. To upgrade, publish and test the new fork commit,
check it out in the submodule, rebuild its wheel, update the image lock, then
commit the gitlink and matching lock together. Runtime jobs use the baked wheel;
putting the submodule on `PYTHONPATH` would bypass the tested image pin.

## Offline build

`packages.lock.json` records the base image, source commits, package filenames,
versions, sizes and hashes. `prepare.py` rejects an uninitialized or dirty Lhotse
checkout, a gitlink/lock mismatch, and any incorrect package hash. It publishes
the staging directory only after all inputs pass.

The recorded wheels and FFmpeg archive are available under:

`/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/outputs/pipeline-runtime-integration-20260914/packages/`

Use those exact inputs for the recorded build. For a new Lhotse pin, build its
pure Python wheel from the initialized clean submodule:

```sh
AUDIO_PACKAGES=/absolute/path/to/offline-packages
export AUDIO_PACKAGES
(
  cd src/3rdparty/lhotse
  export SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
  python -m pip wheel --no-deps --no-build-isolation --wheel-dir "$AUDIO_PACKAGES" .
)
```

The lock refers to the recorded wheel bytes. Rebuilding with a different build
toolchain can change wheel metadata and its hash; record the new input and
validate it before use. The rebuilt pinned wheel in this campaign matched all
380 Lhotse package files in the previously validated wheel.

Stage a fresh build directory and launch the existing Enroot recipe on a
compute host, outside the container:

```sh
python scripts/envs/image/prepare.py --packages "$AUDIO_PACKAGES" --out "$AUDIO_BUILD_OUT"
python scripts/envs/image/launch.py --job-id "$AUDIO_JOB_ID" --out "$AUDIO_BUILD_OUT" \
  --name build --script scripts/envs/image/build.sh --cpus 16 --minutes 45
```

Set `AUDIO_BUILD_OUT` to a new absolute output path and `AUDIO_JOB_ID` to your
allocation. The build verifies the base hash and offline inputs, preserves the
NVIDIA Torch/CUDA/cuDNN stack, compares Python/system inventories, checks added
packages' dependency closure, and exports a new SquashFS. It refuses to overwrite
an existing image. `resume_export.sh` can resume export after installation has
passed verification. The `update_lhotse_*` scripts record the earlier revision-2
campaign operation; use a fresh build for new dependency sets.

TorchAudio was built from `pytorch/audio` v2.11.0 commit
`34c52a67e8941bbd8e6adaca0eb0b9eabec11d78` inside the repaired image with
`USE_CUDA=1`, `TORCH_CUDA_ARCH_LIST=9.0`, and RNNT, alignment and CUDA CTC
extensions enabled. Its wheel targets CUDA 13.3 on GH200. The rebuild reuses
that validated binary; it adds Polars and its ARM64 runtime at version 1.44.2.

## Runtime and validation

Active profile: `scripts/envs/nemo_26_08_audio_tokenization.toml`.
Image: `outputs/pipeline-runtime-integration-20260914/image/nemo-audio-tokenization-26.08.aarch64.sqsh`
under the canonical audio workspace. SHA256:
`cbfa22f5b019af9fde7bf19c7aa9d8983aa18a87970a123fc8adbbe55cd9020f`.

The exported image passed all 682 pipeline tests with no dependency overlay,
CUDA resampling and compiled filtering, cuDNN, WAV/FLAC/MP3/Opus decoding, and
SoXR resampling. The build inventory contains nine added wheels and no removed
packages, changed existing versions, or system-package changes.

The actual Suno and AMI launchers ran in bounded steps of allocation 3397131,
using fresh output paths and baked packages. Music conversion retained both
real input clips; tokenization produced 6,937 tokens for the 173.44-second clip.
The 649-second clip was excluded by the smoke configuration's 600-second
per-clip limit. It was not an inference failure. All twelve AMI turns retained
their IDs, text and timeline metadata through conversion and tokenization,
producing 2,384 audio tokens (including audio boundary markers) and 371 text
tokens. Both normal gap-aware materialization and a separate larger-gap packing
check matched independent token-sequence references exactly. Runtime error and
skip counters were zero; the pre-sampler duration filter is separate from those
counters.

SoXR remains the CPU resampler. Conversion decodes once, resamples channels
sequentially and releases the decoded source after each cut. Conversion thread
limits apply to baked and legacy runtimes. The pipeline's precision settings
are unchanged. The prior old/new runtime comparison found 20 differing codes
among 13,890 with the original cuDNN TF32 setting; disabling cuDNN TF32 in both
runtimes matched that cohort but changed existing token IDs. Keep image and
precision settings fixed when continuing an existing cache. This small-run
certification is not a multi-node throughput or peak-memory guarantee.

The stage helper selects the image on `srun`. Submit with plain `sbatch` from
this checkout or export its absolute `REPO_DIR`; create the chosen Slurm log
directory first. Set `PIPELINE_ENVIRONMENT=/absolute/profile.toml` to test a
candidate across all stages. The older `TOKENIZE_ENVIRONMENT` override still
applies only to tokenization. `LHOTSE_RUNTIME_MODE=legacy` with
`LEGACY_ENVIRONMENT`, `LHOTSE_DIR` and matching dependency inputs explicitly
selects the legacy setup. The earlier images and outputs remain available;
standalone jobs outside `scripts/slurm/` are outside this migration.

Evidence, manifests, exact commands and logs:

`/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/outputs/pipeline-runtime-integration-20260914/`
