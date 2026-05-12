# SFT Audio Tokenization

SFT audio datasets use the same three pipeline stages as the rest of the repo:

1. `convert`: embedded-audio parquet -> SHAR
2. `tokenize`: SHAR -> reusable audio-token cache
3. `materialize`: SFT conversations + audio-token cache -> Megatron indexed data

Dataset cards should stay small. For a normal parquet-backed SFT dataset, use:

```yaml
defaults:
  - /recipe/parquet_sft_audio@_here_
  - /tokenizer/apertus_wavtok@_here_
  - _self_

source:
  path: /path/to/processed/sft/DATASET/media/audio
  files: "audio-*.parquet"

columns:
  audio: audio
  duration: header.duration_sec
  id: audio_id

outputs:
  shar_dir: /capstor/.../audio-datasets/SHAR/sft/DATASET_audio
  tokenized_dir: /capstor/.../audio-datasets/tokenized/sft
  sft_dir: /capstor/.../audio-datasets/tokenized/sft/DATASET
  name: DATASET_audio

tokenizer:
  path: /capstor/.../tokenizer/apertus_emu3.5_wavtok_instruct

materialization:
  sft:
    conversations_dir: /path/to/processed/sft/DATASET/examples
    max_seq_len: 262144
    seq_threshold: 8192
```

Do not put `audio_text_format`, `audio_text_task`, or `audio_cache` knobs in
dataset cards. The `parquet_sft_audio` recipe maps the SFT path to the internal
audio-cache tokenizer mode.

## Conversation Input

SFT conversation parquets need:

- `sample_id`
- `audio_ids`
- a message column, default `messages`

`messages` may be a list of `{role, content}` objects, or `messages_json` may be
a JSON string column. Override with:

```yaml
materialization:
  sft:
    messages_column: messages_json
```

Audio placement supports two explicit forms:

- Literal placeholders in message text, for example `"<audio>\nTranscribe this."`
- Structured message attachments, for example `message.audio = [{"audio_id": "..."}]`

Rows with only top-level `audio_ids` and no message-level placement fail loudly.

## Slurm

Use one launcher for all SFT audio datasets:

```bash
sbatch --reservation=SD-69241-apertus-1-5-0 --time=04:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=288 \
  scripts/slurm/sft_audio.slurm sft/teleantifraud_matching convert

sbatch --reservation=SD-69241-apertus-1-5-0 --time=04:00:00 \
  --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=72 \
  scripts/slurm/sft_audio.slurm sft/teleantifraud_matching tokenize

sbatch --reservation=SD-69241-apertus-1-5-0 --time=02:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=288 \
  scripts/slurm/sft_audio.slurm sft/teleantifraud_matching materialize
```

For another dataset card, replace `sft/teleantifraud_matching` with the dataset
name, for example `sft/marco_longspeech`.

If submitting from inside an existing Pyxis/container allocation, unset the
inherited Pyxis environment override before `sbatch`:

```bash
env -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment \
  sbatch ... scripts/slurm/sft_audio.slurm sft/teleantifraud_matching tokenize
```

Submitting from a normal login shell does not need this.
