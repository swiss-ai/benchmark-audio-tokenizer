# NeMo 26.08 WavTokenizer image

This recipe extends the repaired NeMo 26.08 ARM64 image with the pinned
Lhotse fork, TorchAudio built for CUDA 13.3 / GH200, five dependency wheels,
and FFmpeg 7.1.1. `packages.lock.json` records the exact inputs and hashes.
It supports tokenizing existing SHAR data. The recovered Lhotse sources do
not include the custom atomic preparation APIs.

The build runs through Enroot on a compute host. It verifies the base image
and offline packages, preserves the NVIDIA Torch/CUDA/cuDNN stack, records
package inventories, and exports to a new SquashFS. The runtime profile
must expose `/opt/venv/bin` and `/opt/audio-runtime/ffmpeg/bin`.

The validated TorchAudio source build used `pytorch/audio` v2.11.0 commit
`34c52a67e8941bbd8e6adaca0eb0b9eabec11d78`, inside the repaired image with
`USE_CUDA=1`, `TORCH_CUDA_ARCH_LIST=9.0`, and RNNT, alignment, and CUDA CTC
extensions enabled. The build used `pip wheel --no-deps --no-build-isolation`
and saved the wheel; jobs do not compile or install TorchAudio at startup.

For a fresh build, stage the packages listed in the lock file and their
`SHA256SUMS` under `OUT/packages`, copy the lock to `OUT/package-manifest.json`,
and copy this recipe directory to `OUT/recipe`. Then run from this checkout:

```sh
python scripts/envs/image/launch.py --job-id JOB_ID --out OUT --name build \
  --script scripts/envs/image/build.sh --cpus 16 --minutes 45
```

`resume_export.sh` resumes an interrupted metadata-copy/export stage only
after the installer has passed verification. `update_lhotse_export.sh` is
the recorded revision-2 operation for this campaign: it updates the Lhotse
wheel in the retained isolated rootfs and exports a separate image. Its
parent directory must contain the original `build-work-path.txt`.

Runtime validation must use the exported image, not the retained rootfs.
`validate_runtime.py` checks installed module origins, CUDA operations,
cuDNN, WAV/FLAC/MP3/Opus decoding and soxr. The sampler/SHAR tests run from
a copied `test/` tree so they import the installed Lhotse wheel.

Campaign artifacts, logs, manifests and real-data comparison:

`/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/outputs/nemo-audio-image-20260913/`

`/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/.cache/lhotse-nemo2608-sync-20260913/real-data-comparison/`

The build's failed initial copy and version-check attempts are retained as
evidence. Follow the final revision's validation report for acceptance.

## Validation and activation status (2026-09-14)

Final artifact: `outputs/nemo-audio-image-20260913/revision-2/nemo-audio-tokenization-26.08.aarch64.sqsh`
under the canonical audio workspace. SHA256:
`142c2baf8855318d9f5a23f1b7a7aba13d6c0fe19758b65b61ace8a2e804dd0d`.

The exported image passes CUDA resampling/compiled filtering/cuDNN,
WAV/FLAC/MP3/Opus decoding, and 608 installed-wheel sampler/SHAR/indexing
checks (2 skipped, 13 expected failures). The actual stage/runtime helpers
passed an isolated one-rank, one-worker, three-clip tokenize run, including
a 94-second final clip, with all 4,219 output tokens read back and zero
errors/skips. Inherited installation requests were forced off; module
origins point to the baked packages. This validates the helpers and image,
not a full multi-node corpus run or the missing preparation APIs.

The controlled old/new eager comparison used the same 11 decoded clips,
five batches, checkpoint and GH200. Under the original cuDNN TF32 setting,
20 of 13,890 retained audio codes differ. Disabling only cuDNN TF32 in both
runtimes yields exact code and binary agreement on this cohort. It changes
17 positions from the old TF32-enabled baseline and 13 from the new one.
The pipeline's precision policy has not been changed by this image/launcher
patch. Use the original image/settings when exact continuation of an
existing token cache is required. For new caches, the recommended policy
is this pinned image plus explicit FP32 math; adoption needs a separate
cache identity. Do not interpret small-cohort parity as a corpus guarantee.

The 23 dataset launchers now select their environment on `srun`. Tokenize
selects `scripts/envs/nemo_26_08_audio_tokenization.toml`; convert/materialize
retain `nemo_25_11_audio_legacy.toml`. Submit from this complete checkout or
export its absolute `REPO_DIR`. Use plain `sbatch` with inherited EDF/Pyxis
submission settings cleaned, and create the selected log directory before
submission. Resources, dataset overrides and eager mode are preserved.
The standalone preparation/normalization jobs outside `scripts/slurm/`
are outside this migration. The changes are prepared on an isolated local
branch; no shared default profile or production jobs have been switched.

Detailed manifests and controlled numerical traces are under
`.cache/lhotse-nemo2608-sync-20260913/real-data-comparison/` in the canonical
workspace; `outputs/nemo-audio-image-20260913/` holds image inventories,
regression XML/logs and launcher validation evidence.
