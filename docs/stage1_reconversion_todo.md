# Stage-1 Convert + Tokenize TODO

Goal: rebuild stage-1 SHARs with canonical conversion metadata, including
`custom.rms_db`, then tokenize each dataset only after its converted SHAR passes
validation.

Policy:

- Do not mutate `cut.id`.
- Store conversion metadata in `cut.custom`.
- Keep timestamps first-class where available: preserve `global_offset_sec` and
  interleave metadata such as `clip_start` and `clip_duration`.
- Convert into a new output path first; only promote after canary conversion,
  SHAR validation, canary tokenization, and full tokenization pass.
- Use the unified stage entrypoint for new rebuilds: each stage is its own
  Slurm job because convert is CPU/IO-bound, tokenize is GPU-distributed
  (typically 4 ranks), and materialize is CPU-only. Run them in order:
  `python -m audio_tokenization run dataset=<name> stage=convert`, then
  `stage=tokenize`, then `stage=materialize`. Tokenization reads
  `convert.output.shar_dir` automatically when `tokenize.input_shar_dir: null`.
- Treat `rms_db` status below as sampled evidence. Re-check every dataset before
  launching the full job.

## Pre-Launch Cleanup

- [x] Remove active post-conversion interleave metadata patching.
- [x] Remove active post-conversion RMS metadata patching.
- [x] Remove the legacy `prepare.common` compatibility facade; tests now import
  focused prepare modules directly.
- [x] Add canonical `configs/pipeline/dataset/stage1/*.yaml` specs for
  known-source stage-1 rebuilds: Suno S1, VoxPopuli VAD, and UPS all-lang VAD.
- [ ] Add canonical specs for AudioSet and MTG-Jamendo after locating or
  reconstructing their converters.
- [x] Add canonical Gemeinderat spec using the raw WDS MP3 tar shards.
- [x] Remove the old standalone `audio_tokenization.tokenize` entrypoint,
  dataset config tree, and launchers; rebuilds now go through
  `python -m audio_tokenization run ...`.
- [x] Update known external stage-1 launchers to write versioned rebuild paths
  and use canonical `/capstor/.../raw` inputs.

## Datasets

| Order | Dataset | Current stage-1 path | Size | Sampled `custom.rms_db` | Known source | Known launcher | Current tokenized output | Action |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| 1 | AudioSet balanced | `/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/audioset_bal_train_audio_only` | 5.5G | Missing | `/capstor/store/cscs/swissai/infra01/audio-datasets/raw/javisdata-audios/audio/AudioSet` | TBD | `audioset_bal_train_audio_only_dur2-200`, `audioset_bal_train_shar_lhotse_audio_only_dur2-200` | Find/modernize converter, convert canary, tokenize canary, full convert, full tokenize. |
| 2 | Gemeinderat | `/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/gemeinderat_audio_only` | 142G | Missing | `/capstor/store/cscs/swissai/infra01/audio-datasets/raw/Gemeinderat/shards` | `configs/pipeline/dataset/stage1/gemeinderat.yaml` | `gemeinderat_shar_lhotse_audio_only_dur1-200` | Canary passed on 2 raw shards; run full convert, full validate, full tokenize. |
| 3 | MTG-Jamendo train | `/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/mtg_jamendo_train` | 391G | Missing | `/capstor/store/cscs/swissai/infra01/audio-datasets/raw/mtg-jamendo/data/train` | TBD | `mtg_jamendo_train_S1_audio_only_dur5-200` | Find/modernize converter, convert canary, tokenize canary, full convert, full tokenize. |
| 4 | AudioSet unbalanced | `/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/audioset_unbal_train_audio_only` | 781G | Missing | `/capstor/store/cscs/swissai/infra01/audio-datasets/raw/javisdata-audios/audio/AudioSet` | TBD | `audioset_unbal_train_shar_lhotse_audio_only_dur2-200` | Reuse balanced converter path after the small balanced canary succeeds. |
| 5 | Suno S1 | `/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/suno_s1_shar` | 2.0T | Missing | `/capstor/store/cscs/swissai/infra01/audio-datasets/raw/suno/shards_s1` | `/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/suno/prepare_to_shar.slurm` | `suno_train_S1_audio_only_dur5-200` | Convert S1 only to a new path, verify `rms_db`, then tokenize. |
| 6 | VoxPopuli VAD | `/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/voxpopuli_shar` | 27T | Missing | `/capstor/store/cscs/swissai/infra01/audio-datasets/raw/voxpopuli/raw_audios` plus VAD metadata | `/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/voxpopuli/prepare_vad_to_shar.slurm` | `voxpopuli_vad_audio_only_dur5-200` | Verify raw/VAD paths, run canary, then full convert/tokenize. |
| 7 | UPS all-lang VAD | `/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/ups_all_lang_shar_v2` | 30T | Missing | `/capstor/store/cscs/swissai/infra01/audio-datasets/raw/unsupervised_peoples_speech_commercial_wds/audio*` plus VAD metadata | `/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/peoples_speech/unsupervised/prepare_vad_to_shar.slurm` | `ups_all_lang_vad_v2_audio_only_dur5-200` | Run after smaller converters prove the pipeline; this is the high-cost full rebuild. |
| 8 | UPS per-lang | `/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/ups_per_lang` | 30T | Likely missing, verify | Same UPS raw plus per-language VAD metadata | `/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/peoples_speech/unsupervised/prepare_vad_to_shar_per_lang.slurm` | TBD | Do not rebuild until we confirm it is still consumed; it appears to be an alternate UPS view. |
| 9 | CommonVoice | `/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/commonvoice` | 310G | Present in sampled manifest | `/capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24` | `/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/commonvoice24/download_and_prepare_commonvoice_shar.sh` | `commonvoice_ca_de_en_es_fr_it_zh-CN_S1_audio_only_dur3-500` | Verify coverage across languages; rebuild only if metadata is inconsistent or config changed. |

## Execution Board

Use this table to decide what to do next. A dataset is not ready for full
conversion until source/launcher, canary conversion, canary validation, and
canary tokenization are all checked.

Current next step: run a Suno S1 canary through the unified graph, because its
source and config are known. In parallel, resolve the smaller AudioSet balanced
converter so it can become the next canary.

| Order | Dataset | Source + launcher resolved | Pipeline spec | Canary convert | Canary SHAR validate | Canary tokenize | Full convert | Full SHAR validate | Full tokenize | Config switch | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | AudioSet balanced | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | Find or reconstruct converter. |
| 2 | Gemeinderat | [x] | [x] | [x] | [x] | [x] | [ ] | [ ] | [ ] | [ ] | Run full conversion with `dataset=stage1/gemeinderat`. |
| 3 | MTG-Jamendo train | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | Find or reconstruct converter. |
| 4 | AudioSet unbalanced | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | Reuse AudioSet balanced converter after canary passes. |
| 5 | Suno S1 | [x] | [x] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | Run canary conversion with `dataset=stage1/suno_s1`. |
| 6 | VoxPopuli VAD | [x] | [x] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | Update array launcher to new output root, then run canary node. |
| 7 | UPS all-lang VAD | [x] | [x] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | Run only after smaller canaries prove conversion + tokenization. |
| 8 | UPS per-lang | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | Confirm whether this view is still consumed. |
| 9 | CommonVoice | [x] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | Sample all selected languages for `rms_db` before deciding rebuild. |

## Per-Dataset Checklist

- [ ] Identify the exact source path, VAD/manifest path, and current conversion launcher.
- [ ] Run a tiny canary conversion into a new temporary SHAR path.
- [ ] Verify the converted cuts include `custom.rms_db` and preserve timestamp metadata.
- [ ] Run SHAR validation on the canary.
- [ ] Run canary tokenization into a new temporary tokenized path.
- [ ] Launch full conversion into a new versioned stage-1 path.
- [ ] Validate full SHAR output.
- [ ] Launch full tokenization into a new versioned tokenized path.
- [ ] Update dataset config only after conversion and tokenization pass.
- [ ] Keep old outputs until downstream training config and counts are checked.

## Canary Notes

- Gemeinderat canary used raw shards `gemeinderat-000000.tar` and
  `gemeinderat-000001.tar` with `shard_size=2000`.
- Conversion read 7,248 raw MP3 clips, wrote 7,221 SHAR cuts, skipped 27 quiet
  clips, and produced 4 cut shards covering 117.4 hours. All converted cuts had
  `custom.rms_db`; manifest schema v2 reported RMS coverage 7,221/7,221.
- Tokenization with `min_duration=1.0` processed 7,187 cuts; the remaining 34
  converted cuts were shorter than 1.0s. No max-duration, RMS, sample-rate, or
  tokenizer errors were observed.
- 1-GPU and 4-GPU audio-only outputs matched by cut-id/token set:
  7,187 cuts, 16,901,591 tokens, max token drift 0.

## Open Work

- Locate or reconstruct the conversion launchers for AudioSet and MTG-Jamendo
  before launching full jobs.
- Decide whether `ups_per_lang` is still a production input or only an alternate
  view of `ups_all_lang_shar_v2`.
- Verify the runtime resampling backend on the Slurm nodes before large launches.
- Choose final versioned output names for rebuilt SHAR and tokenized datasets.
