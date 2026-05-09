# benchmark-audio-tokenizers

A comprehensive benchmarking framework for evaluating discrete audio tokenizer performance across different models and datasets.

## Overview

This repository provides tools and scripts for systematically evaluating audio tokenizers on multilingual speech data. The project runs on **Clariden (CSCS Alps)** and supports multiple datasets (EuroSpeech, FLEURS, GTZAN, NatureLM) with automatic dataset detection and unified evaluation pipeline.

## Project Goals

The benchmarking framework focuses on two main objectives:

1. **Statistical Evaluation**: Compute comprehensive metrics (MSE, SNR, SDR, PESQ, STOI, ESTOI) on 100 samples per language to assess tokenizer performance statistically.

2. **Sample Generation**: Generate 5 audio samples per tokenizer-dataset-language combination for listening evaluation and qualitative assessment.

## Prerequisites

- Access to **Clariden (CSCS Alps)** cluster
- Assignment to the `infra01` group with proper `.edf` configuration (recommended)
- `uv` package manager (recommended, for creating virtual environments)
- PyTorch NGC 24.11 environment (recommended)

## Production Conversion Defaults

Use `conversion.resampling_backend: soxr` for large SHAR conversion runs. `soxr`
is a dedicated CPU resampler with high-quality output, predictable performance,
and less container/ABI coupling than torchaudio. Decode can still dominate
runtime on compressed audio, so the end-to-end speedup is dataset-dependent, but
`soxr` is the preferred production default for resampling.

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd benchmark-audio-tokenizer
```

### 2. Set Up Virtual Environments

**Important:** Virtual environments should be created within the NGC 24.11 environment on Clariden.

The project uses `uv` for fast virtual environment management. We use a two-stage dependency compilation approach:

- **Top-level dependencies** (`requirements-*-topdeps.txt`): High-level packages specified by the user
- **Sub-dependencies** (`requirements-*-subdeps.txt`): All transitive dependencies compiled by `uv pip compile`

This approach allows us to use system-installed PyTorch from NGC (avoiding CUDA compatibility issues) and install dependencies in a controlled, reproducible manner

#### Create All Tokenizer Environments

```bash
# Make sure you're in NGC 24.11 environment
# Then create all venvs:
make venvs
```

This creates virtual environments for all tokenizers:
- `.venv-neucodec/`
- `.venv-cosyvoice2/`
- `.venv-xcodec2/`
- `.venv-wavtokenizer/`

#### Alternatively: Create Individual Environments

```bash
# Create a specific tokenizer environment
make neucodec      # CPU-only PyTorch
make cosyvoice2    # Uses system-site-packages for PyTorch
make xcodec2       # CPU-only PyTorch
make wavtokenizer  # Uses system-site-packages for PyTorch
```

Each Makefile target:
1. Removes the old venv (if exists)
2. Creates a new venv with `uv`
3. Compiles top-level dependencies to sub-dependencies (where applicable)
4. Removes conflicting PyTorch entries from compiled dependencies
5. Installs dependencies without overshadowing system PyTorch from NGC
6. Verifies the installation

### 3. Verify Setup with Example Notebooks

Before running evaluations, we recommend testing your setup with the example notebooks in the `examples/` directory:

```bash
# Activate a tokenizer environment
source .venv-neucodec/bin/activate

# Start Jupyter
jupyter notebook examples/neucodec.ipynb
```

Available notebooks:
- `neucodec.ipynb`
- `cosyvoice2.ipynb`
- `xcodec2.ipynb`
- `wavtokenizer.ipynb`

These notebooks demonstrate basic tokenizer usage and help verify that your environment is correctly configured.

## Project Structure

```
.
├── examples/                      # Example notebooks for testing tokenizers
├── logs/                          # Execution logs (.out and .err files per job)
├── metrics/                       # Evaluation results and metrics (JSON output)
├── samples/                       # Generated audio samples for listening evaluation
├── scripts/                       # All Python scripts and shell scripts
│   ├── tokenizer_evaluation.py    # Main evaluation script
│   ├── generate_samples.py        # Sample generation script
│   ├── submit_missing_jobs.py     # Automatic job submission
│   ├── analyze_tokenizers.py      # Analysis and visualization
│   └── ...
├── src/
│   ├── audio_tokenizers/          # Tokenizer implementations and wrappers
│   └── repos/                     # External repository dependencies
├── .venv-*/                       # Virtual environments for each tokenizer
├── requirements-*-topdeps.txt     # Top-level dependencies
├── requirements-*-subdeps.txt     # Compiled sub-dependencies
└── Makefile                       # Environment setup automation
```

## Datasets

**EuroSpeech:**
- 22 languages: bosnia-herzegovina, bulgaria, croatia, denmark, estonia, finland, france, germany, greece, iceland, italy, latvia, lithuania, malta, norway, portugal, serbia, slovakia, slovenia, sweden, uk, ukraine

**FLEURS:**
- 40 languages configured (102 available)

**GTZAN:**
- 10 music genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- less than 100 samples each

**NatureLM:**
- 6 audio datasets: Xeno-canto, WavCaps, NatureLM, Watkins, iNaturalist, Animal Sound Archive

**Total Coverage:** 78+ languages/datasets across 4 dataset types

## Tokenizers Evaluated

- ✅ NeuCodec
- ✅ XCodec2
- ✅ CosyVoice2
- ✅ WavTokenizer

## Running Evaluations

### Automatic Job Submission

The recommended approach is to use `submit_missing_jobs.py` to automatically detect and submit missing tokenizer-language combinations.

#### Step 1: Dry Run

Always start with a dry run to see what would be submitted:

```bash
python scripts/submit_missing_jobs.py --dry-run
```

This shows:
- Which tokenizer-language combinations are missing
- How jobs would be grouped (by dataset or language)
- What commands would be executed

#### Step 2: Test with One Job

Before submitting all missing jobs, test with a single submission:

```bash
python scripts/submit_missing_jobs.py --submit-one
```

This submits only one job per task (metrics and samples) to verify everything works correctly.

#### Step 3: Submit All Missing Jobs

Once verified, submit all missing combinations:

```bash
# Submit both metrics and samples (default)
python scripts/submit_missing_jobs.py

# Or submit only one task
python scripts/submit_missing_jobs.py --task metrics
python scripts/submit_missing_jobs.py --task samples
```

#### Important Options

**Validation (`--validate-metrics`):**
- Validates that metrics JSON files are complete and have all required fields with values
- Invalid files are treated as missing and will be re-submitted
- **Why needed:** Sometimes jobs fail partially, creating incomplete JSON files. This ensures only complete results are considered.

```bash
python scripts/submit_missing_jobs.py --validate-metrics
```

**Grouping (`--group-by`):**
- `dataset` (default): Groups missing languages by dataset, creating one job per tokenizer-dataset combination
  - Fewer jobs, longer runtime per job
  - More efficient for cluster resource usage
- `language`: Creates one job per tokenizer-language combination
  - More jobs, shorter runtime per job
  - Better for fine-grained control and faster individual completions

```bash
# Group by dataset (default, recommended)
python scripts/submit_missing_jobs.py --group-by dataset

# Group by language
python scripts/submit_missing_jobs.py --group-by language
```

**Prerequisites for Job Submission:**
- You must be assigned to the `infra01` group
- Your `.edf` file must be properly configured for SLURM
- The script automatically checks for running jobs to avoid duplicates

### Manual Evaluation

You can also run evaluations manually:

```bash
source .venv-neucodec/bin/activate

# Single language
python scripts/tokenizer_evaluation.py --tokenizer neucodec --language germany

# Multiple languages
python scripts/tokenizer_evaluation.py --tokenizer neucodec --languages germany en_us ja_jp

# Entire dataset
python scripts/tokenizer_evaluation.py --tokenizer neucodec --dataset eurospeech
```

### Results Organization

Results are automatically organized:

```
metrics/
├── neucodec_eurospeech_germany_results.json      # Per-language results
├── neucodec_fleurs_en_us_results.json            # Per-language results
└── ...

samples/
├── neucodec/
│   ├── eurospeech/
│   │   └── germany/
│   │       ├── metadata.json
│   │       └── sample_*.wav
│   └── fleurs/
│       └── en_us/
│           ├── metadata.json
│           └── sample_*.wav
└── ...
```

Each metrics file includes:
- Language name and dataset origin
- All metrics (MSE, SNR, SDR, PESQ, STOI, ESTOI) with mean, std, min, max, median
- Tokenization statistics (tokens per second, compression ratio)
- Number of samples evaluated
- Might be incomplete for certain metrics

## Analysis and Visualization

After collecting results, use `analyze_tokenizers.py` to generate comprehensive analysis and visualizations.

### Setup Analysis Environment

Create a simple virtual environment with standard packages for analysis:

```bash
uv venv .venv-analysis
source .venv-analysis/bin/activate
uv pip install pandas matplotlib seaborn numpy
```

### Run Analysis

```bash
source .venv-analysis/bin/activate
python scripts/analyze_tokenizers.py
```

The script automatically:
- Detects all tokenizers from result files in `metrics/`
- Validates metrics files for completeness
- Generates comprehensive visualizations
- Creates summary statistics

### Generated Outputs

All outputs are saved to the `results/` directory:

**Visualizations:**
- `language_coverage.png` - Heatmap showing which languages each tokenizer has (with metrics completeness)
- `overall_comparison.png` - Performance comparison across all languages (may not be fair if tokenizers tested different languages)
- `common_languages_comparison.png` - Fair comparison using only languages all tokenizers have
- `metric_comparison_bars.png` - Bar charts comparing mean performance by metric
- `dataset_comparison.png` - Performance breakdown by dataset
- `compression_efficiency.png` - Compression ratio and tokens per second analysis
- `correlation_heatmap.png` - Correlation matrix between metrics
- `top_bottom_languages_*.png` - Top and bottom performing languages for key metrics
- `scatter_*_vs_*.png` - Scatter plots comparing metric relationships

**Statistics:**
- `analysis_summary.txt` - Comprehensive text summary including:
  - Aggregation methodology explanation
  - Language coverage analysis
  - Per-tokenizer statistics
  - Overall comparisons
  - Fair comparisons (common languages only)

**Key Features:**
- Automatically handles missing or incomplete metrics files
- Shows metrics completeness (0-6 valid metrics per language)
- Provides both overall and fair comparisons
- Explains aggregation methodology (language-weighted vs sample-weighted)

## Evaluation Metrics

The framework computes comprehensive reconstruction quality metrics:

### Reconstruction Quality
- **MSE** (Mean Squared Error): Measures reconstruction error
- **SNR** (Signal-to-Noise Ratio): Overall signal quality in dB
- **SDR** (Signal-to-Distortion Ratio): Distortion measurement in dB
- **PESQ**: Perceptual Evaluation of Speech Quality (1.0-4.5 scale)
- **STOI**: Short-Time Objective Intelligibility (0-1 scale)
- **ESTOI**: Extended STOI for improved accuracy

### Tokenization Efficiency
- **Tokens per second**: Tokenization rate
- **Compression ratio**: Original size / Token size

All metrics include: mean, standard deviation, min, max, and median values computed across 100 samples per language.

## Troubleshooting

### Environment Setup
- **"uv: command not found"**: Install `uv` package manager
- **PyTorch CUDA issues**: Ensure you're using NGC 24.11 environment
- **Dependency conflicts**: The Makefile handles PyTorch conflicts automatically by using system-site-packages where needed

### Job Submission
- **"Permission denied"**: Verify you're in the `infra01` group and `.edf` is configured
- **Jobs not submitting**: Check SLURM configuration and cluster status
- **Duplicate jobs**: The script automatically checks for running jobs, but verify with `squeue`

### Evaluation
- **"Language not found"**: Verify language code spelling and dataset download status
- **"Dataset not found at PATH"**: Check cache directories
- **PESQ errors**: Audio is automatically resampled to 16kHz, check audio quality
- **Memory issues**: Adjust memory requirements in `submit_missing_jobs.py` or use SLURM with more memory

## Contributing

This project is part of the Data Science Lab course at ETH Zurich, autumn semester 2025.

## Citation

If you use this benchmarking framework, please cite the relevant datasets.
