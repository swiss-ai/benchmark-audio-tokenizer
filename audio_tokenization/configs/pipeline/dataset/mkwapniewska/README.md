# Reality-gap Polish ASR configs

Synthetic Polish speech from VoxCPM-2 (speaker 1636), tokenized for the Apertus
audio LLM. Five source corpora, all parquet input + WavTokenizer-40 + Apertus
omni-tokenizer. All outputs are isolated under `audio-datasets/reality_gap/` so
nothing here mixes with the production SHAR/tokenized trees.

| config                       |    utts |  hours | parquets | size  |
|------------------------------|--------:|-------:|---------:|------:|
| `pl_cv_spk1636`              |  23,510 |   32.8 |       24 |  5.6G |
| `pl_mls_spk1636`             |  21,913 |   90.6 |       28 | 15.6G |
| `pl_granary_yodas_spk1636`   |   3,239 |    6.6 |        4 |  1.1G |
| `pl_granary_ytc_spk1636`     |   2,647 |    8.9 |        4 |  1.5G |
| `pl_granary_1kh_spk1636`     | 164,573 |  850.3 |      189 | 146.5G |

`pl_granary_yodas_spk1636` is the smallest — use it first for plumbing checks.

## Output layout

```
/capstor/store/cscs/swissai/infra01/audio-datasets/reality_gap/
├── SHAR/voxcpm2/<subset>/        # written by `convert`
└── tokenized/voxcpm2/<subset>/   # written by `tokenize`
```

Both paths are hard-coded in each config's `outputs` block — no override needed.

## How to run

### Easiest: chained submission (convert then tokenize)

```bash
scripts/slurm/mkwapniewska_pl_chain.sh pl_granary_yodas_spk1636
```

That submits `convert` (CPU) and queues `tokenize` (GPU) with
`--dependency=afterok` so tokenize only runs if convert succeeds.

For the big `pl_granary_1kh_spk1636` (850 h) raise the walltimes:

```bash
CONVERT_TIME=04:00:00 TOKENIZE_TIME=08:00:00 \
  scripts/slurm/mkwapniewska_pl_chain.sh pl_granary_1kh_spk1636
```

### Manual: one stage at a time

```bash
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment  # if inside an interactive container

# convert (CPU)
sbatch --reservation=SD-69241-apertus-1-5-0 --time=02:00:00 \
       --nodes=1 --ntasks=1 --cpus-per-task=288 \
       scripts/slurm/mkwapniewska_pl.slurm pl_granary_yodas_spk1636 convert

# tokenize (GPU) — only after convert succeeds
sbatch --reservation=SD-69241-apertus-1-5-0 --time=04:00:00 \
       --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=72 \
       scripts/slurm/mkwapniewska_pl.slurm pl_granary_yodas_spk1636 tokenize
```

## Logs & success markers

Both stages stream to `logs/mk_pl_<JOBID>.{out,err}`. Stage success is signalled
by a `_SUCCESS` file in the corresponding output directory:

```
reality_gap/SHAR/voxcpm2/<subset>/_SUCCESS         # convert OK
reality_gap/tokenized/voxcpm2/<subset>/transcribe/voxcpm2_<subset>/_SUCCESS  # tokenize OK
```

`stats_summary.json` in the tokenize output gives per-rank sample counts and
throughput.

## Validation reference

`pl_granary_yodas_spk1636` validated end-to-end on 2026-05-18:
* convert: 45 s, 3,239 cuts written, 0 errors.
* tokenize: 4 ranks × 24 s, 941 K audio + 111 K text tokens, 0 errors.
* Whisper-large-v3 PL WER on original synth audio: **6.78 %**.
* WER after WavTokenizer-40 encode + decode: **28.08 %** (+21.30 abs pts).

The 21-pt codec gap is the round-trip degradation from the 40-tok/s audio
codec on Polish. It is the starting baseline for the real-vs-synth ablation
on this branch — nothing in these configs introduces it.
