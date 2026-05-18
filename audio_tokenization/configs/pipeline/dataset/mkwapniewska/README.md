# Reality-gap Polish ASR configs

| config                       |    utts |  hours | parquets | size  |
|------------------------------|--------:|-------:|---------:|------:|
| `pl_cv_spk1636`              |  23,510 |   32.8 |       24 |  5.6G |
| `pl_mls_spk1636`             |  21,913 |   90.6 |       28 | 15.6G |
| `pl_granary_yodas_spk1636`   |   3,239 |    6.6 |        4 |  1.1G |
| `pl_granary_ytc_spk1636`     |   2,647 |    8.9 |        4 |  1.5G |
| `pl_granary_1kh_spk1636`     | 164,573 |  850.3 |      189 | 146.5G |

## Patched Lhotse

Pipeline needs the patched fork (atomic SHAR + fixed sampler). Upstream Lhotse
silently corrupts SHAR.

```bash
git clone -b fix/duration-batcher-and-shar-reader \
  https://github.com/Alvorecer721/lhotse.git /capstor/scratch/cscs/mkwapniewska/dev/lhotse
export LHOTSE_DIR=/capstor/scratch/cscs/mkwapniewska/dev/lhotse
```

> ⚠️ `scripts/utils/source_lhotse_runtime.sh` line 15 hardcodes the default
> `LHOTSE_DIR` to **Yixuan's scratch** (`/iopsstor/scratch/cscs/xyixuan/dev/lhotse`).
> That path can vanish at any time. **Always `export LHOTSE_DIR=...` to your own
> checkout** before submitting — sbatch will propagate the env into the job.

The slurm launchers source that script for you; it honours `LHOTSE_DIR`,
installs torchaudio/torchcodec into `/opt/venv`, and sets `OMP_NUM_THREADS=1`
for `convert`. For interactive Python:

```bash
INSTALL_TORCHAUDIO=1 INSTALL_TORCHCODEC=1 \
  source scripts/utils/source_lhotse_runtime.sh
```

## Run

```bash
scripts/slurm/mkwapniewska_pl_chain.sh pl_granary_yodas_spk1636
```

That submits `convert` (`--ntasks=1 --cpus-per-task=288`, all 288 CPUs on one
task) and chains `tokenize` (`--ntasks-per-node=4 --gpus-per-node=4
--cpus-per-task=72`, one task per GPU) with `afterok`.
For the big one:

```bash
CONVERT_TIME=04:00:00 TOKENIZE_TIME=08:00:00 \
  scripts/slurm/mkwapniewska_pl_chain.sh pl_granary_1kh_spk1636
```

Outputs land in:
```
reality_gap/SHAR/voxcpm2/<subset>/                                     # convert
reality_gap/tokenized/voxcpm2/<subset>/transcribe/voxcpm2_<subset>/    # tokenize
```
Success = `_SUCCESS` file in each. Throughput on GH200 is ~10–20K audio
tokens/s/GPU; `stats_summary.json` in the tokenize dir has per-rank numbers.
Logs: `logs/mk_pl_<JOBID>.{out,err}`.

## Validated reference (2026-05-18, `pl_granary_yodas_spk1636`)

* convert 45 s, tokenize 4×24 s, 941 K audio + 111 K text tokens, 0 errors.
* Whisper-large-v3 PL WER on original synth: **6.78 %**.
* WER after WavTokenizer round-trip: **28.08 %** (+21.30 pts codec gap —
  baseline degradation, not introduced by these configs).
