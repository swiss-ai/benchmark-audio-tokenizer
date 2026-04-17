#!/usr/bin/env python3
"""Backward-compatible facade for shared prepare_data helpers.

New code should import from the focused modules in this package directly.
"""

from __future__ import annotations

from audio_tokenization.prepare.audio_ops import (
    apply_audio_pipeline,
    build_recording_from_audio_bytes,
    make_rms_filter_fn,
    normalize_batch_peak,
    rms_db,
    should_skip_quiet,
    to_mono,
)
from audio_tokenization.prepare.cli import (
    add_audio_processing_args,
    add_external_metadata_args,
    add_language_arg,
    add_parallelism_args,
    add_shar_output_args,
    add_text_tokenizer_args,
)
from audio_tokenization.prepare.columnar import (
    _get_field,
    _projected_columns,
    add_columnar_metadata_args,
    extract_row_metadata,
)
from audio_tokenization.prepare.constants import (
    MIN_RMS_DB,
    PREPARE_STATE_FILE,
    PREPARE_SUMMARY_FILE,
    SUCCESS_MARKER_FILE,
    WORKER_ASSIGNMENT_FILE,
    WORKER_STATS_FILE,
    MetadataEntry,
    _MISSING,
)
from audio_tokenization.prepare.identity import (
    add_input_clip_id_parser_arg,
    assign_universal_ids,
    resolve_input_source_and_clip_num,
    set_universal_cut_id,
)
from audio_tokenization.prepare.metadata import (
    load_external_metadata,
    lookup_external_metadata,
    normalize_optional_path,
    resolve_sample_text_and_custom,
)
from audio_tokenization.prepare.runtime import (
    audio_md5,
    build_audio_index,
    build_shar_index,
    build_shar_index_from_parts,
    check_worker_reuse,
    distribute_round_robin,
    ensure_worker_assignment,
    init_worker_process,
    load_worker_assignment,
    mark_partition_success,
    run_aggregate,
    run_pool_and_finalize,
    setup_partition_dir,
    validate_or_write_prepare_state,
    write_worker_assignment,
    write_worker_result,
)
from audio_tokenization.prepare.text_ops import (
    load_text_tokenizer,
    make_text_tokenize_fn,
)

