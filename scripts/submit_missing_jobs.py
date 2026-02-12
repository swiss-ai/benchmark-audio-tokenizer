#!/usr/bin/env python3
"""
Automatically submit missing tokenizer-language combination jobs.

Detects missing combinations in metrics/ and samples/ and automatically
submits the corresponding SLURM jobs with the correct venv.

Usage:
    python scripts/submit_missing_jobs.py --task metrics
    python scripts/submit_missing_jobs.py --task samples
    python scripts/submit_missing_jobs.py --task both  # default
    python scripts/submit_missing_jobs.py --dry-run  # only show, don't submit
    python scripts/submit_missing_jobs.py --local  # execute locally instead of submitting via SLURM
    python scripts/submit_missing_jobs.py --group-by language  # group by language instead of dataset
    python scripts/submit_missing_jobs.py --tokenizers xcodec2 cosyvoice2  # only submit for specific tokenizers
    python scripts/submit_missing_jobs.py --languages germany en_us  # only submit for specific languages
    python scripts/submit_missing_jobs.py --submit-one  # only submit one job per task
    python scripts/submit_missing_jobs.py --validate-metrics  # also validate file completeness
"""

import glob
import subprocess
import shlex
import json
import time
from pathlib import Path
from typing import Set, Tuple, List, Dict, Optional
import argparse

# Import dataset configs from existing scripts
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Tokenizer configurations
TOKENIZERS = [
    # 'neucodec',
    # 'xcodec2',
    # 'cosyvoice2',
    # 'wavtokenizer',
    # 'glm4voice',
    'flexicodec'
]

# Dataset configurations (shared with tokenizer_evaluation.py and generate_samples.py)
DATASETS = {
    'eurospeech': {
        'languages': [
            "bosnia-herzegovina", "bulgaria", "croatia", "denmark", "estonia", "finland",
            "france", "germany", "greece", "iceland", "italy", "latvia", "lithuania",
            "malta", "norway", "portugal", "serbia", "slovakia", "slovenia",
            "sweden", "uk", "ukraine"
        ],
        'expected_samples_per_language': 100  # Standard limit, but some languages may have fewer
    },
    'fleurs': {
        'languages': [
            "ast_es", "ca_es", "nl_nl", "en_us", "gl_es", "hu_hu", "ga_ie", 
            "kea_cv", "lb_lu", "oc_fr", "es_419", "cy_gb",
            "hy_am", "be_by", "cs_cz", "ka_ge", "mk_mk", "pl_pl", "ro_ro", "ru_ru",
            "cmn_hans_cn", "yue_hant_hk", "ja_jp", "ko_kr",
            "hi_in", "bn_in", "ta_in", "te_in",
            "th_th", "vi_vn", "id_id",
            "af_za", "sw_ke", "am_et", "yo_ng",
            "ar_eg", "tr_tr", "he_il", "fa_ir"
        ],
        'expected_samples_per_language': 100  # Standard limit, but some languages may have fewer
    },
    "naturelm": {
        "languages": [
            "Xeno-canto", "WavCaps", "NatureLM", "Watkins",
            "iNaturalist", "Animal Sound Archive"
        ],
        'expected_samples_per_language': 100  # Standard limit, but some sources may have fewer
    },
    "gtzan": {
        "languages": [
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"
        ],
        'expected_samples_per_language': 25  # GTZAN was downloaded with 25 samples per genre
    },
}

# Base paths
PROJECT_ROOT = project_root
METRICS_DIR = PROJECT_ROOT / "metrics"
SAMPLES_DIR = PROJECT_ROOT / "samples"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# SLURM job template
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%j_{job_name}.out
#SBATCH --error={log_dir}/%j_{job_name}.err
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal

# Disable core dumps to prevent filling up home directory
ulimit -c 0

# Set OpenBLAS threads to match CPU count (prevents warnings)
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Store start time as Unix timestamp for duration calculation
START_TIME_SEC=$(date +%s)
START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')

# Log job start timestamp
echo "==========================================" >&2
echo "Job started at: $START_TIME_READABLE" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
echo "Job name: {job_name}" >&2
echo "==========================================" >&2
echo "=========================================="
echo "Job started at: $START_TIME_READABLE"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: {job_name}"
echo "=========================================="

source {venv_path}
cd {project_root}

# Function to log job end timestamp and duration
log_job_end() {{
    EXIT_CODE=$?
    END_TIME_SEC=$(date +%s)
    END_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Calculate duration
    DURATION_SEC=$((END_TIME_SEC - START_TIME_SEC))
    DURATION_HOURS=$((DURATION_SEC / 3600))
    DURATION_MINS=$(((DURATION_SEC % 3600) / 60))
    DURATION_SECS=$((DURATION_SEC % 60))
    DURATION_FORMATTED=$(printf "%02d:%02d:%02d" $DURATION_HOURS $DURATION_MINS $DURATION_SECS)
    
    echo "==========================================" >&2
    echo "Job ended at: $END_TIME_READABLE" >&2
    echo "Duration: $DURATION_FORMATTED (${{DURATION_SEC}}s)" >&2
    echo "Exit code: $EXIT_CODE" >&2
    echo "==========================================" >&2
    echo "=========================================="
    echo "Job ended at: $END_TIME_READABLE"
    echo "Duration: $DURATION_FORMATTED (${{DURATION_SEC}}s)"
    echo "Exit code: $EXIT_CODE"
    echo "=========================================="
}}

# Set trap to log job end timestamp on exit
trap log_job_end EXIT

{command}
"""


def get_dataset_for_language(language: str) -> Tuple[str, None]:
    """Determine which dataset a language belongs to."""
    for dataset_name, config in DATASETS.items():
        if language in config['languages']:
            return dataset_name, None
    return None, None


def extract_language_from_filename(filename: str, tokenizer: str) -> str:
    """
    Extract language from filename, handling different naming conventions.
    Same logic as in analyze_tokenizers.py, but removes fleurs_ prefix to match DATASETS config.
    """
    basename = Path(filename).stem  # Remove extension
    
    # Remove the tokenizer prefix
    if basename.startswith(f"{tokenizer}_"):
        basename = basename[len(tokenizer)+1:]
    
    # Remove _results suffix
    if basename.endswith("_results"):
        basename = basename[:-8]
    
    # Remove dataset prefixes (eurospeech, fleurs, gtzan, naturelm)
    known_datasets = ['eurospeech', 'fleurs', 'gtzan', 'naturelm']
    for dataset in known_datasets:
        if basename.startswith(f"{dataset}_"):
            basename = basename[len(dataset)+1:]
    
    return basename


def get_expected_combinations() -> Set[Tuple[str, str]]:
    """Get all expected tokenizer-language combinations."""
    combinations = set()
    
    # Get all languages from all datasets
    all_languages = []
    for dataset_config in DATASETS.values():
        all_languages.extend(dataset_config['languages'])
    
    # Create all combinations
    for tokenizer in TOKENIZERS:
        for language in all_languages:
            combinations.add((tokenizer, language))
    
    return combinations


def get_missing_by_dataset(missing_combinations: Set[Tuple[str, str]]) -> Dict[Tuple[str, str], List[str]]:
    """
    Group missing combinations by (tokenizer, dataset) and return list of languages.
    
    Returns:
        Dict mapping (tokenizer, dataset) -> list of languages
    """
    grouped = {}
    for tokenizer, language in missing_combinations:
        dataset_name, _ = get_dataset_for_language(language)
        if dataset_name:
            key = (tokenizer, dataset_name)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(language)
    return grouped


def validate_metrics_file(file_path: Path) -> bool:
    """
    Validate that a metrics file is complete and has all required fields with values.
    
    Returns True if the file is valid, False otherwise.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False
    
    # Check required top-level fields
    required_fields = ['language', 'dataset', 'num_samples', 'metrics', 
                       'tokens_per_second', 'compression_ratio']
    for field in required_fields:
        if field not in data:
            return False
    
    # Check num_samples is valid
    if not isinstance(data['num_samples'], int) or data['num_samples'] <= 0:
        return False
    
    # Check if num_samples is complete (not significantly less than expected)
    dataset_name = data.get('dataset')
    if dataset_name and dataset_name in DATASETS:
        expected_samples = DATASETS[dataset_name].get('expected_samples_per_language')
        if expected_samples is not None:
            # Allow 10% tolerance or at least 5 samples difference to account for failures
            # But if num_samples is less than 80% of expected, consider it incomplete
            tolerance = max(5, int(expected_samples * 0.1))
            min_acceptable = expected_samples - tolerance
            if data['num_samples'] < min_acceptable:
                return False  # Significantly fewer samples than expected (likely incomplete)
    
    # Check metrics structure
    metrics = data.get('metrics', {})
    if not isinstance(metrics, dict):
        return False
    
    # Expected metric names
    expected_metrics = ['mse', 'snr_db', 'sdr_db', 'pesq', 'stoi', 'estoi']
    
    # Check that all expected metrics exist and are not None (treat null metrics as incomplete)
    for metric_name in expected_metrics:
        if metric_name not in metrics:
            return False  # Metric is missing
        if metrics[metric_name] is None:
            return False  # Metric is null (incomplete)
        
        # If metric exists and is not None, validate its structure
        metric_data = metrics[metric_name]
        if not isinstance(metric_data, dict):
            return False
        # Check required fields for non-null metrics
        required_metric_fields = ['mean', 'std', 'min', 'max', 'median']
        for field in required_metric_fields:
            if field not in metric_data:
                return False
            # Check that values are numbers (not None)
            if metric_data[field] is None:
                return False
    
    # Check tokens_per_second structure
    tps = data.get('tokens_per_second', {})
    if not isinstance(tps, dict) or 'mean' not in tps or 'std' not in tps:
        return False
    if tps['mean'] is None or tps['std'] is None:
        return False
    
    # Check compression_ratio structure
    cr = data.get('compression_ratio', {})
    if not isinstance(cr, dict) or 'mean' not in cr or 'std' not in cr:
        return False
    if cr['mean'] is None or cr['std'] is None:
        return False
    
    return True


def get_existing_metrics(validate: bool = False) -> Set[Tuple[str, str]]:
    """
    Get all existing tokenizer-language combinations from metrics/.
    
    Args:
        validate: If True, also validate that metrics files are complete
                  and have all required fields with values.
    """
    existing = set()
    
    if not METRICS_DIR.exists():
        return existing
    
    # Find all result files
    result_files = glob.glob(str(METRICS_DIR / "*_*_results.json"))
    
    for file in result_files:
        basename = Path(file).name
        if 'summary' in basename or 'all_results' in basename:
            continue
        
        # Extract tokenizer (first part before underscore)
        if '_' in basename:
            tokenizer = basename.split('_')[0]
            language = extract_language_from_filename(basename, tokenizer)
            
            # If validation is enabled, check that the file is valid
            if validate:
                file_path = Path(file)
                if not validate_metrics_file(file_path):
                    continue  # Skip invalid files
            
            existing.add((tokenizer, language))
    
    return existing


def get_existing_samples() -> Set[Tuple[str, str]]:
    """Get all existing tokenizer-language combinations from samples/."""
    existing = set()
    
    if not SAMPLES_DIR.exists():
        return existing
    
    # Check for tokenizer directories
    for tokenizer_dir in SAMPLES_DIR.iterdir():
        if not tokenizer_dir.is_dir():
            continue
        
        tokenizer = tokenizer_dir.name
        
        # Check for dataset directories
        for dataset_dir in tokenizer_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            # Check for language directories
            for lang_dir in dataset_dir.iterdir():
                if not lang_dir.is_dir():
                    continue
                
                language = lang_dir.name
                
                # Check if metadata.json exists (indicates samples were generated)
                metadata_file = lang_dir / "metadata.json"
                if metadata_file.exists():
                    existing.add((tokenizer, language))
    
    return existing


def get_venv_path(tokenizer: str) -> str:
    """Get the venv path for a tokenizer."""
    venv_path = PROJECT_ROOT / f".venv-{tokenizer}" / "bin" / "activate"
    return str(venv_path)


def get_venv_python(tokenizer: str) -> str:
    """Get the Python interpreter path from the venv."""
    python_path = PROJECT_ROOT / f".venv-{tokenizer}" / "bin" / "python"
    return str(python_path)


def sanitize_job_name(tokenizer: str, identifier: str, task: str) -> str:
    """
    Create a safe job name for SLURM.
    
    Args:
        tokenizer: Tokenizer name
        identifier: Language name or dataset name
        task: Task name (metrics or samples)
    """
    safe = identifier.replace(' ', '_').replace('/', '_').replace('-', '_')
    safe = ''.join(c if c.isalnum() or c in '_-' else '_' for c in safe)
    return f"{task}_{tokenizer}_{safe}"


def is_job_running(tokenizer: str, identifier: str, task: str) -> bool:
    """
    Check if a SLURM job for this combination is already running or pending.
    
    Args:
        tokenizer: Tokenizer name
        identifier: Language name or dataset name
        task: Task name (metrics or samples)
    
    Returns True if a job with the same name is found in the queue.
    """
    job_name = sanitize_job_name(tokenizer, identifier, task)
    
    try:
        # Check for jobs with this name in the queue (running, pending, etc.)
        result = subprocess.run(
            ['squeue', '-n', job_name, '-h', '--format=%i'],
            capture_output=True,
            text=True,
            check=False,
            timeout=5  # Timeout to avoid hanging
        )
        # If there's any output, a job is running
        return len(result.stdout.strip()) > 0
    except subprocess.TimeoutExpired:
        # If squeue times out, assume no job is running to be safe
        return False
    except Exception:
        # If squeue fails for any reason, assume no job is running
        # (better to submit a duplicate than to miss a job)
        return False


def submit_job(tokenizer: str, language: str = None, dataset: str = None, languages: List[str] = None, task: str = None, dry_run: bool = False, local: bool = False) -> Tuple[bool, str]:
    """
    Submit a SLURM job for a tokenizer-language or tokenizer-dataset combination.
    
    Args:
        tokenizer: Tokenizer name
        language: Language name (for language-based jobs)
        dataset: Dataset name (for dataset-based jobs)
        languages: List of language names (for dataset-based jobs with specific languages)
        task: Task name (metrics or samples)
        dry_run: If True, only show what would be submitted
        local: If True, execute locally instead of submitting via SLURM
    
    Returns:
        (success, status) tuple where:
        - success: True if job was submitted or would be submitted (dry-run)
        - status: 'submitted', 'skipped', 'failed', or 'completed' (for local execution)
    """
    # Validate arguments: either language, or (dataset with optional languages list)
    if language is not None and (dataset is not None or languages is not None):
        raise ValueError("Cannot specify both language and dataset/languages")
    if language is None and dataset is None:
        raise ValueError("Either language or dataset must be provided")
    
    if task is None:
        raise ValueError("Task must be provided")
    
    venv_path = get_venv_path(tokenizer)
    
    if not Path(venv_path).exists():
        print(f"  ⚠ Warning: venv not found: {venv_path}")
        return (False, 'failed')
    
    # For local execution, also check if Python interpreter exists
    venv_python = None
    if local:
        venv_python = get_venv_python(tokenizer)
        if not Path(venv_python).exists():
            print(f"  ⚠ Warning: venv Python interpreter not found: {venv_python}")
            return (False, 'failed')
    
    # Create identifier for job name
    if languages:
        identifier = dataset if dataset else f"{len(languages)}_languages"
    else:
        identifier = dataset if dataset else language
    job_name = sanitize_job_name(tokenizer, identifier, task)
    
    # Check if a job for this combination is already running (only for SLURM jobs)
    if not dry_run and not local and is_job_running(tokenizer, identifier, task):
        print(f"  ⊘ Skipped (job already running): {tokenizer} - {identifier} ({task})")
        return (False, 'skipped')
    
    # Determine command based on task
    tokenizer_quoted = shlex.quote(tokenizer)
    
    if task == "metrics":
        if languages:
            # Use --languages with all missing languages (only process these, not entire dataset)
            languages_quoted = ' '.join(shlex.quote(lang) for lang in languages)
            command = f"python scripts/tokenizer_evaluation.py --tokenizer {tokenizer_quoted} --languages {languages_quoted}"
        elif dataset:
            command = f"python scripts/tokenizer_evaluation.py --tokenizer {tokenizer_quoted} --dataset {shlex.quote(dataset)}"
        else:
            command = f"python scripts/tokenizer_evaluation.py --tokenizer {tokenizer_quoted} --language {shlex.quote(language)}"
    elif task == "samples":
        if languages:
            # Use --languages with all missing languages (only process these, not entire dataset)
            languages_quoted = ' '.join(shlex.quote(lang) for lang in languages)
            command = f"python scripts/generate_samples.py --tokenizer {tokenizer_quoted} --languages {languages_quoted} --num-samples 5"
        elif dataset:
            command = f"python scripts/generate_samples.py --tokenizer {tokenizer_quoted} --datasets {shlex.quote(dataset)} --num-samples 5"
        else:
            command = f"python scripts/generate_samples.py --tokenizer {tokenizer_quoted} --languages {shlex.quote(language)} --num-samples 5"
    else:
        print(f"  ✗ Unknown task: {task}")
        return (False, 'failed')
    
    # Create SLURM script
    slurm_script = SLURM_TEMPLATE.format(
        job_name=job_name,
        log_dir=str(LOGS_DIR),
        venv_path=venv_path,
        project_root=str(PROJECT_ROOT),
        command=command
    )
    
    if dry_run:
        if local:
            print(f"  [DRY RUN] Would execute locally: {tokenizer} - {identifier} ({task})")
        else:
            print(f"  [DRY RUN] Would submit: {tokenizer} - {identifier} ({task})")
        print(f"    Job name: {job_name}")
        print(f"    Command: {command}")
        print()
        return (True, 'submitted')
    
    # Local execution
    if local:
        try:
            print(f"  ▶ Executing locally: {tokenizer} - {identifier} ({task})")
            print(f"    Command: {command}")
            
            # Replace 'python' with the venv Python interpreter path
            # This handles 'python scripts/...' format
            if command.startswith('python '):
                command_with_venv = command.replace('python ', f'{venv_python} ', 1)
            else:
                # Fallback: prepend the venv Python
                command_with_venv = f"{venv_python} {command}"
            
            # Execute directly (blocking, sequential execution)
            result = subprocess.run(
                command_with_venv,
                shell=True,
                cwd=str(PROJECT_ROOT),
                check=False  # Don't raise on non-zero exit, we'll check it
            )
            
            if result.returncode == 0:
                print(f"  ✓ Completed: {tokenizer} - {identifier} ({task})")
                return (True, 'completed')
            else:
                print(f"  ✗ Failed with exit code {result.returncode}: {tokenizer} - {identifier} ({task})")
                return (False, 'failed')
        except Exception as e:
            print(f"  ✗ Failed to execute locally: {e}")
            return (False, 'failed')
    
    # Submit via sbatch
    try:
        result = subprocess.run(
            ['sbatch'],
            input=slurm_script,
            text=True,
            capture_output=True,
            check=True
        )
        job_id = result.stdout.strip().split()[-1]
        print(f"  ✓ Submitted job {job_id}: {tokenizer} - {identifier} ({task})")
        time.sleep(1)  # Wait 1 second between submissions
        return (True, 'submitted')
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to submit job: {e}")
        print(f"    stderr: {e.stderr}")
        return (False, 'failed')


def main():
    parser = argparse.ArgumentParser(
        description="Automatically submit missing tokenizer-language combination jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit only missing metrics jobs
  python scripts/submit_missing_jobs.py --task metrics

  # Submit only missing sample generation jobs
  python scripts/submit_missing_jobs.py --task samples

  # Both tasks (default)
  python scripts/submit_missing_jobs.py --task both

  # Dry-run: Only show what would be submitted, don't actually submit
  python scripts/submit_missing_jobs.py --dry-run

  # Execute jobs locally instead of submitting via SLURM (useful on compute nodes)
  python scripts/submit_missing_jobs.py --local

  # Submit only one job per task (useful after dry-run to test step by step)
  python scripts/submit_missing_jobs.py --submit-one

  # Validate metrics files for completeness (treats invalid files as missing)
  python scripts/submit_missing_jobs.py --validate-metrics

  # Group jobs by dataset (fewer jobs, longer runtime per job)
  python scripts/submit_missing_jobs.py --group-by dataset

  # Only for specific tokenizers
  python scripts/submit_missing_jobs.py --tokenizers xcodec2 cosyvoice2

  # Only for specific languages
  python scripts/submit_missing_jobs.py --languages germany en_us
        """
    )
    
    parser.add_argument(
        '--task',
        choices=['metrics', 'samples', 'both'],
        default='both',
        help='Which tasks to submit: metrics, samples, or both (default: both)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only show which jobs would be submitted, without actually submitting'
    )
    
    parser.add_argument(
        '--validate-metrics',
        action='store_true',
        help='Validate that metrics files are complete and have all required fields with values. '
             'Invalid files will be treated as missing. (default: only check file existence)'
    )
    
    parser.add_argument(
        '--group-by',
        choices=['language', 'dataset'],
        default='dataset',
        help='Group jobs by language (one job per language) or dataset (one job per dataset, all languages together). Default: dataset'
    )
    
    parser.add_argument(
        '--tokenizers',
        nargs='+',
        default=None,
        help='Only submit for specific tokenizers (default: all)'
    )
    
    parser.add_argument(
        '--languages',
        nargs='+',
        default=None,
        help='Only submit for specific languages (default: all)'
    )
    
    parser.add_argument(
        '--submit-one',
        action='store_true',
        help='Only submit one job per task (useful after dry-run to test step by step)'
    )
    
    parser.add_argument(
        '--local',
        action='store_true',
        help='Execute jobs locally instead of submitting via SLURM (useful when running on a compute node)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("AUTOMATIC SUBMISSION OF MISSING JOBS")
    print("="*60)
    print(f"Task(s): {args.task}")
    print(f"Group by: {args.group_by}")
    print(f"Dry-run: {args.dry_run}")
    if args.local:
        print(f"Local execution: Yes (executing jobs locally)")
    if args.submit_one:
        print(f"Submit one per task: Yes (only first job per task)")
    if args.validate_metrics:
        print(f"Validate metrics: Yes (checking completeness)")
    if args.tokenizers:
        print(f"Tokenizers: {', '.join(args.tokenizers)}")
    if args.languages:
        print(f"Languages: {', '.join(args.languages)}")
    print("="*60)
    
    # Get expected and existing combinations
    print("\n1. Analyzing expected combinations...")
    expected = get_expected_combinations()
    print(f"   Expected combinations: {len(expected)}")
    
    # Filter by tokenizers if specified
    if args.tokenizers:
        invalid_tokenizers = [t for t in args.tokenizers if t not in TOKENIZERS]
        if invalid_tokenizers:
            print(f"  ⚠ Warning: Invalid tokenizers: {', '.join(invalid_tokenizers)}")
        expected = {(t, l) for t, l in expected if t in args.tokenizers}
        print(f"   After tokenizer filter: {len(expected)}")
    
    # Filter by languages if specified
    if args.languages:
        expected = {(t, l) for t, l in expected if l in args.languages}
        print(f"   After language filter: {len(expected)}")
    
    # Check existing files
    missing_metrics = set()
    missing_samples = set()
    
    if args.task in ['metrics', 'both']:
        validation_note = " (with validation)" if args.validate_metrics else ""
        print(f"\n2. Checking existing metrics results{validation_note}...")
        existing_metrics = get_existing_metrics(validate=args.validate_metrics)
        print(f"   Existing combinations: {len(existing_metrics)}")
        missing_metrics = expected - existing_metrics
        print(f"   Missing combinations: {len(missing_metrics)}")
    
    if args.task in ['samples', 'both']:
        print("\n3. Checking existing samples...")
        existing_samples = get_existing_samples()
        print(f"   Existing combinations: {len(existing_samples)}")
        missing_samples = expected - existing_samples
        print(f"   Missing combinations: {len(missing_samples)}")
    
    # Submit missing jobs
    total_submitted = 0
    total_failed = 0
    total_skipped = 0
    
    if args.task in ['metrics', 'both'] and missing_metrics:
        if args.group_by == 'dataset':
            print(f"\n4. Submitting missing metrics jobs grouped by dataset ({len(missing_metrics)} languages)...")
            missing_by_dataset = get_missing_by_dataset(missing_metrics)
            print(f"   Grouped into {len(missing_by_dataset)} dataset jobs")
            if args.submit_one:
                print(f"   [Submit-one mode: Only submitting first job]")
            for (tokenizer, dataset), languages in sorted(missing_by_dataset.items()):
                print(f"   {tokenizer} - {dataset}: {len(languages)} languages")
                success, status = submit_job(tokenizer, dataset=dataset, languages=languages, task="metrics", dry_run=args.dry_run, local=args.local)
                if status in ['submitted', 'completed']:
                    total_submitted += 1
                    if args.submit_one:
                        break  # Only submit one job
                elif status == 'skipped':
                    total_skipped += 1
                    if args.submit_one:
                        break  # Only submit one job
                else:
                    total_failed += 1
        else:
            print(f"\n4. Submitting missing metrics jobs ({len(missing_metrics)})...")
            if args.submit_one:
                print(f"   [Submit-one mode: Only submitting first job]")
            for tokenizer, language in sorted(missing_metrics):
                success, status = submit_job(tokenizer, language=language, task="metrics", dry_run=args.dry_run, local=args.local)
                if status in ['submitted', 'completed']:
                    total_submitted += 1
                    if args.submit_one:
                        break  # Only submit one job
                elif status == 'skipped':
                    total_skipped += 1
                    if args.submit_one:
                        break  # Only submit one job
                else:
                    total_failed += 1
    
    if args.task in ['samples', 'both'] and missing_samples:
        if args.group_by == 'dataset':
            print(f"\n5. Submitting missing sample jobs grouped by dataset ({len(missing_samples)} languages)...")
            missing_by_dataset = get_missing_by_dataset(missing_samples)
            print(f"   Grouped into {len(missing_by_dataset)} dataset jobs")
            if args.submit_one:
                print(f"   [Submit-one mode: Only submitting first job]")
            for (tokenizer, dataset), languages in sorted(missing_by_dataset.items()):
                print(f"   {tokenizer} - {dataset}: {len(languages)} languages")
                success, status = submit_job(tokenizer, dataset=dataset, languages=languages, task="samples", dry_run=args.dry_run, local=args.local)
                if status in ['submitted', 'completed']:
                    total_submitted += 1
                    if args.submit_one:
                        break  # Only submit one job
                elif status == 'skipped':
                    total_skipped += 1
                    if args.submit_one:
                        break  # Only submit one job
                else:
                    total_failed += 1
        else:
            print(f"\n5. Submitting missing sample jobs ({len(missing_samples)})...")
            if args.submit_one:
                print(f"   [Submit-one mode: Only submitting first job]")
            for tokenizer, language in sorted(missing_samples):
                success, status = submit_job(tokenizer, language=language, task="samples", dry_run=args.dry_run, local=args.local)
                if status in ['submitted', 'completed']:
                    total_submitted += 1
                    if args.submit_one:
                        break  # Only submit one job
                elif status == 'skipped':
                    total_skipped += 1
                    if args.submit_one:
                        break  # Only submit one job
                else:
                    total_failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if args.dry_run:
        if args.local:
            print(f"Jobs that would be executed: {total_submitted}")
        else:
            print(f"Jobs that would be submitted: {total_submitted}")
    else:
        if args.local:
            print(f"Successfully completed: {total_submitted}")
        else:
            print(f"Successfully submitted: {total_submitted}")
        if total_skipped > 0:
            print(f"Skipped (already running): {total_skipped}")
        if total_failed > 0:
            print(f"Failed: {total_failed}")
    print("="*60)


if __name__ == "__main__":
    main()

