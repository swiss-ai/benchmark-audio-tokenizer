# Pipeline Configs

Run a single stage for one dataset:

```bash
python -m audio_tokenization run dataset=sft/teleantifraud_matching stage=tokenize
```

Inspect all configured stages without running:

```bash
python -m audio_tokenization plan dataset=cooldown/ami
```

`config.yaml` intentionally requires `dataset=...`; there is no default dataset.
`run` and `clean` also require `stage=convert|tokenize|materialize`.
`plan` and `status` inspect all stages when `stage` is unset.

Dataset configs live under:

- `dataset/sft/` - supervised speech/audio conversation datasets
- `dataset/cooldown/` - cooldown speech/text corpora
- `dataset/stage1/` - large audio-only pretraining rebuilds
- `dataset/stage2/` - stage-2 conversion specs
- `dataset/internal/` - restricted-distribution internal datasets

These directories are only human/operator categories. They do not change runtime
behavior. The actual pipeline behavior comes from the dataset YAML contents:
the selected `recipe`, source fields, output fields, and enabled stages.

`lct/` is not a dataset category. It is a materialization output bucket produced
from stage-2 or cooldown-style datasets when `seq_threshold` routing is enabled.

Adding a dataset:

```bash
cp audio_tokenization/configs/pipeline/dataset/sft/_template.yaml \
   audio_tokenization/configs/pipeline/dataset/sft/my_dataset.yaml
python -m audio_tokenization plan dataset=sft/my_dataset
```

Keep `recipe/` flat. A recipe defines the pipeline shape, such as
`parquet_audio_only`, `parquet_audio_text_direct`, or `parquet_sft_audio`.
Keep dataset files concrete: source paths, columns, output paths, and only the
overrides that differ from the recipe.
