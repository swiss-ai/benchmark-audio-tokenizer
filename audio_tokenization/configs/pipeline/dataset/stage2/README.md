# Stage 2 Dataset Configs

This directory is for current runnable or rebuildable stage-2 dataset specs.
It is not a complete historical manifest of every source that appears in the
merged Apertus 1.5 stage-2 training product.

Current runnable specs:

- `libriheavy_large.yaml`

The authoritative merged stage-2 tokenized product lives at:

```text
/capstor/store/cscs/swissai/infra01/audio-datasets/Apertus1p5_stage2_tokenized/MERGED
```

As checked on 2026-05-12, `MERGED` contains these source/product names.

## ASR

```text
aishell134
coral_v3_read_aloud
eurospeech
f1_team_radio
gigaspeech
gigaspeech2_th_id_vi_direct
granary_yodas
granary_ytc_asr
kathbath
kazakh_speech
legco_speech
libriheavy_large
mls_7lang
omnilingual_asr
parlament_parla
parlaspeech_rs
peoples_speech
seamless_align
spc_r
vimedcss
voxpopuli
wenetspeech
zeroth_korean
zoengjyutgaai
```

## AST

```text
granary_yodas
granary_ytc_ast
seamless_align
voxpopuli_ast_sampled
```

## Interleaved

```text
aishell4
eurospeech
granary_yodas
legco_speech
libriheavy_large
mls_7lang
parlaspeech_rs
peoples_speech
spc_r
vimedcss
voxpopuli
wenetspeech
zeroth_korean
zoengjyutgaai
```

`asr`, `ast`, and `intlv` are output-product prefixes in `MERGED`, not nested
config categories. A single source may produce more than one product.

`lct/` is also not a dataset category. It is a materialization output bucket for
sequences routed above a configured `seq_threshold`.
