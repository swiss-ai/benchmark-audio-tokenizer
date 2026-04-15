"""Shared CLI argument builders for prepare_data scripts."""

from __future__ import annotations

from pathlib import Path


def add_external_metadata_args(parser, *, include_custom_fields: bool = True) -> None:
    """Add shared CLI options for transcript/custom metadata overrides."""
    parser.add_argument(
        "--external-metadata",
        type=str,
        default=None,
        help=(
            "Path to external metadata file (.tsv, .jsonl/.jsonl.gz, or .csv). "
            "When set, entries are looked up by sample ID and can override text "
            "and provide additional custom fields."
        ),
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default="id",
        help="Key/column name for sample ID in external metadata (default: 'id')",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Key/column name for transcript text in external metadata (default: 'text')",
    )
    if include_custom_fields:
        parser.add_argument(
            "--custom-fields",
            type=str,
            nargs="*",
            default=None,
            help=(
                "Keys/columns to copy from external metadata into cut.custom "
                "(e.g. --custom-fields language speaker)"
            ),
        )


def add_shar_output_args(parser, *, shard_size_default=2000, shar_dir_required=True):
    """Add --shar-dir, --shard-size, --shar-format."""
    parser.add_argument("--shar-dir", type=Path, required=shar_dir_required, default=None,
                        help="Output directory for Shar format")
    parser.add_argument("--shard-size", type=int, default=shard_size_default,
                        help=f"Samples per Shar shard (default: {shard_size_default})")
    parser.add_argument("--shar-format", type=str, default="flac",
                        choices=["flac", "wav", "mp3", "opus"],
                        help="Audio format in Shar (default: flac)")


def add_audio_processing_args(parser, *, target_sr_default=24000,
                              include_min_sr=False, include_mono_downmix=False):
    """Add --target-sr, --resampling-backend, and optionally --min-sr, --no-mono-downmix."""
    parser.add_argument("--target-sr", type=int, default=target_sr_default,
                        help=f"Target sample rate (default: {target_sr_default})")
    parser.add_argument("--resampling-backend", type=str, default="soxr",
                        choices=["torchaudio", "soxr", "ffmpeg"],
                        help="Resampling backend (default: soxr)")
    if include_min_sr:
        parser.add_argument("--min-sr", type=int, default=16000,
                            help="Drop audio below this sample rate (default: 16000)")
    if include_mono_downmix:
        parser.add_argument("--no-mono-downmix", action="store_true",
                            help="Select channel 0 instead of averaging stereo channels")


def add_language_arg(parser):
    """Add --language for setting supervision.language on all cuts."""
    parser.add_argument("--language", type=str, default=None,
                        help="Language tag to set on all supervisions (e.g. fi, en, zh). "
                             "Overridden by --language-column if both are set.")


def add_text_tokenizer_args(parser, *, include_custom_columns=False):
    """Add --text-tokenizer and optionally --text-tokenize-custom-columns."""
    parser.add_argument("--text-tokenizer", type=str, default=None,
                        help="Path to tokenizer.json for pre-tokenizing supervision text")
    if include_custom_columns:
        parser.add_argument("--text-tokenize-custom-columns", type=str, nargs="*",
                            default=None,
                            help="Custom columns to also pre-tokenize. "
                                 "Stored as {col}_tokens in cut.custom.")


def add_parallelism_args(parser, *, num_workers_default=20,
                         include_mp_start_method=False):
    """Add --num-workers and optionally --mp-start-method."""
    parser.add_argument("--num-workers", type=int, default=num_workers_default,
                        help=f"Number of parallel workers (default: {num_workers_default})")
    if include_mp_start_method:
        parser.add_argument("--mp-start-method", type=str, default="forkserver",
                            choices=["fork", "forkserver", "spawn"],
                            help="Multiprocessing start method (default: forkserver)")

