#!/usr/bin/env python3
"""Run small real-data canaries across source format and tokenization mode.

The fixture matrix is intentionally bounded:

* WDS + audio-text direct: Samromur Children dev tar + headered TSV metadata.
* WDS + interleave: one tiny podcast recording tar + generated flat metadata.
* Parquet + audio-text direct/interleave: SPC-R segmented test parquet.

Each tokenization case runs once with 1 GPU and once with 4 GPUs, then compares
the production identity contract:

* Megatron outputs: exact cut-id set equality plus marker-aware audio prefix
  equality, allowing only bounded padding-trim tail drift.
* Interleave cache outputs: exact ``(source_id, clip_num) -> token sequence`` equality.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from audio_tokenization.contracts.artifacts import INTERLEAVE_CACHE_OUTPUT_STEM
from audio_tokenization.utils.indexed_dataset.cut_id_sidecar import compare_cutid_token_sets
from audio_tokenization.utils.io import atomic_write_json
from audio_tokenization.utils.token_mapping import get_structure_tokens

PYTHON = os.environ.get("AUDIO_TOKENIZATION_PYTHON", "/opt/venv/bin/python")
DEV_LHOTSE = os.environ.get("AUDIO_TOKENIZATION_DEV_LHOTSE", "/iopsstor/scratch/cscs/xyixuan/dev/lhotse")
ROOT = Path(os.environ.get("AUDIO_TOKENIZATION_CANARY_ROOT", f"/tmp/audio_tok_raw_matrix_{int(time.time())}"))
FFMPEG_ROOT = Path(
    os.environ.get(
        "AUDIO_TOKENIZATION_FFMPEG_ROOT",
        "/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64",
    )
)
TOKENIZER = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok"
TEXT_TOKENIZER = f"{TOKENIZER}/tokenizer.json"
RUN_STAMP = time.time_ns()
TRIM_TOLERANCE = 5


@dataclass(frozen=True)
class TokenizeCase:
    name: str
    dataset: str
    output_name: str
    output_kind: str
    shar_dir: Path
    overrides: tuple[str, ...]
    materialize: bool = False


def _pipeline_args(args: list[str], *, rank: int, world_size: int, run_id: str) -> list[str]:
    env = {
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": str(rank),
        "AUDIO_TOKENIZATION_RUN_ID": run_id,
        "PATH": f"{FFMPEG_ROOT / 'bin'}:{os.environ.get('PATH', '')}",
    }
    code = (
        "import json, os, sys; "
        f"sys.path.insert(0, {DEV_LHOTSE!r}); "
        f"os.environ.update({env!r}); "
        "from audio_tokenization.__main__ import main; "
        f"main(json.loads({json.dumps(args)!r}))"
    )
    return [PYTHON, "-c", code]


def _run_pipeline(args: list[str], *, log_path: Path, rank: int = 0, world_size: int = 1, run_id: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _pipeline_args(args, rank=rank, world_size=world_size, run_id=run_id)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline command failed; see {log_path}")


def _run_pipeline_multi(args: list[str], *, log_dir: Path, world_size: int, run_id: str) -> list[Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    logs: list[Path] = []
    for rank in range(world_size):
        log_path = log_dir / f"rank_{rank}.log"
        logs.append(log_path)
        cmd = _pipeline_args(args, rank=rank, world_size=world_size, run_id=run_id)
        log = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        procs.append((rank, proc, log))

    failures = []
    for rank, proc, log in procs:
        rc = proc.wait()
        log.close()
        if rc != 0:
            failures.append((rank, rc))
    if failures:
        raise RuntimeError(f"Distributed pipeline command failed: {failures}; see {log_dir}")
    return logs


def _cuda_preflight() -> None:
    result = subprocess.run(
        [
            PYTHON,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {DEV_LHOTSE!r}); "
                "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
            ),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout.strip(), flush=True)
    if result.returncode != 0 or "True 4" not in result.stdout:
        raise RuntimeError("CUDA preflight failed")


def _base_runtime_overrides() -> list[str]:
    return [
        "runtime.resume=false",
        "dataset.tokenization.num_workers=8",
        "dataset.tokenization.prefetch_factor=4",
        "dataset.tokenization.checkpoint_interval_batches=2000",
    ]


def _samromur_overrides(shar_dir: Path, tokenized_dir: Path, output_name: str) -> list[str]:
    raw = "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___language-and-voice-lab___samromur_children"
    return [
        "dataset=stage1/ups_all_lang_vad",
        "dataset.name=canary_samromur_wds_direct",
        "dataset.recipe.mode=audio_text_direct",
        "dataset.source.shards=[" + f"{raw}/corpus/speech/dev.tar.gz" + "]",
        "dataset.source.vad.enabled=false",
        "dataset.source.min_sr=16000",
        f"dataset.columns.external_metadata={raw}/corpus/files/metadata_dev.tsv",
        "dataset.columns.id_field=audio_id",
        "dataset.columns.text_field=normalized_text",
        "dataset.columns.custom_fields=[speaker_id,gender,age,duration]",
        "+dataset.language=is",
        "dataset.conversion.enabled=true",
        f"dataset.conversion.text_tokenizer={TEXT_TOKENIZER}",
        "dataset.conversion.num_workers=4",
        "dataset.conversion.shard_size=2000",
        "dataset.tokenization.min_duration=1.0",
        "dataset.tokenization.max_duration=60.0",
        f"dataset.outputs.shar_dir={shar_dir}",
        f"dataset.outputs.tokenized_dir={tokenized_dir}",
        f"dataset.outputs.name={output_name}",
    ]


def _rchiera_metadata(path: Path) -> Path:
    src = Path(
        "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/"
        "hf___rchiera___podcast-transcripts___downloaded/cleaned/000000617/cuts.jsonl.gz"
    )
    out = path / "rchiera_000000617_metadata.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rt", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            obj = json.loads(line)
            sup = (obj.get("supervisions") or [{}])[0]
            custom = obj.get("custom") or {}
            payload = {
                "id": obj["id"],
                "text": sup.get("text"),
                "global_offset_sec": custom.get("parent_start"),
                "parent_end": custom.get("parent_end"),
                "parent_key": custom.get("parent_key"),
                "segment_idx": custom.get("segment_idx"),
            }
            g.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return out


def _rchiera_overrides(shar_dir: Path, tokenized_dir: Path, output_name: str, metadata_path: Path) -> list[str]:
    raw = "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___rchiera___podcast-transcripts___downloaded/cleaned/000000617"
    return [
        "dataset=stage1/ups_all_lang_vad",
        "dataset.name=canary_rchiera_wds_interleave",
        "dataset.recipe.mode=audio_text_interleaved",
        "dataset.recipe.materialize_interleave=true",
        f"dataset.source.shards=[{raw}/recording.tar]",
        "dataset.source.vad.enabled=false",
        "dataset.source.min_sr=16000",
        f"dataset.columns.external_metadata={metadata_path}",
        "dataset.columns.id_field=id",
        "dataset.columns.text_field=text",
        "dataset.columns.custom_fields=[global_offset_sec,parent_end,parent_key,segment_idx]",
        "dataset.timeline.parser=trailing_number",
        "+dataset.language=en",
        "dataset.conversion.enabled=true",
        f"dataset.conversion.text_tokenizer={TEXT_TOKENIZER}",
        "dataset.conversion.num_workers=1",
        "dataset.conversion.shard_size=2000",
        "dataset.tokenization.min_duration=1.0",
        "dataset.tokenization.max_duration=60.0",
        "dataset.materialization.interleave.enabled=true",
        "dataset.materialization.interleave.max_gap_sec=5.0",
        "dataset.materialization.interleave.transcribe_ratio=null",
        "dataset.materialization.interleave.num_workers=4",
        f"dataset.outputs.shar_dir={shar_dir}",
        f"dataset.outputs.tokenized_dir={tokenized_dir}",
        f"dataset.outputs.name={output_name}",
    ]


def _spc_overrides(shar_dir: Path, tokenized_dir: Path, output_name: str, *, mode: str) -> list[str]:
    materialize_enabled = "true" if mode == "audio_text_interleaved" else "false"
    return [
        "dataset=cooldown/ccpodcasts",
        f"dataset.name=canary_spc_parquet_{mode}",
        f"dataset.recipe.mode={mode}",
        f"dataset.recipe.materialize_interleave={materialize_enabled}",
        f"dataset.source.path=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/spc-r-segmented/test",
        "dataset.source.files=shard_*.parquet",
        "dataset.columns.id=id",
        "dataset.columns.text=text",
        "dataset.columns.duration=duration",
        "dataset.columns.keep=[]",
        "dataset.timeline.parser=spc",
        "dataset.timeline.clip_start=null",
        "dataset.timeline.clip_end=null",
        "dataset.timeline.clip_duration=null",
        "dataset.conversion.enabled=true",
        "dataset.conversion.num_workers=8",
        "dataset.conversion.shard_size=2000",
        "dataset.tokenization.min_duration=1.0",
        "dataset.tokenization.max_duration=60.0",
        "dataset.materialization.max_gap_sec=null",
        "dataset.materialization.transcribe_ratio=null",
        f"dataset.materialization.interleave.enabled={materialize_enabled}",
        "dataset.materialization.interleave.max_gap_sec=null",
        "dataset.materialization.interleave.transcribe_ratio=null",
        "dataset.materialization.interleave.num_workers=8",
        f"dataset.outputs.shar_dir={shar_dir}",
        f"dataset.outputs.tokenized_dir={tokenized_dir}",
        f"dataset.outputs.name={output_name}",
    ]


def _convert_once(name: str, overrides: list[str]) -> None:
    print(f"[convert] {name}", flush=True)
    _run_pipeline(
        ["run", "stage=convert", *overrides],
        log_path=ROOT / "logs" / "convert" / f"{name}.log",
        run_id=f"raw-canary-{RUN_STAMP}-{name}-convert",
    )


def _token_output_dir(case: TokenizeCase, output_root: Path) -> Path:
    if case.output_kind == "megatron":
        return output_root / "transcribe" / case.output_name
    if case.output_kind == "interleave":
        return output_root / INTERLEAVE_CACHE_OUTPUT_STEM / case.output_name
    raise ValueError(f"unknown output kind: {case.output_kind}")


def _tokenize_args(case: TokenizeCase, output_root: Path) -> list[str]:
    return [
        "run",
        "stage=tokenize",
        *_base_runtime_overrides(),
        *case.overrides,
        f"dataset.tokenization.input_shar_dir={case.shar_dir}",
        f"dataset.outputs.tokenized_dir={output_root}",
        f"dataset.outputs.name={case.output_name}",
    ]


def _stats_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "stats_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"missing stats summary: {summary_path}")
    return json.loads(summary_path.read_text())


def _compare_stats(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keys = (
        "samples_processed",
        "audio_tokens",
        "text_tokens",
        "errors",
        "samples_skipped",
        "rms_skipped",
        "no_text_skipped",
    )
    return {
        key: {"1gpu": left.get(key), "4gpu": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    }


def _collect_interleave_cache(root: Path) -> dict[tuple[str, int], tuple[tuple[int, ...], tuple[int, ...], Any, Any]]:
    result = {}
    for clips_path in sorted(root.glob("**/clips.*.parquet")):
        rank_dir = clips_path.parent
        stem = clips_path.name.split(".")[1]
        audio = np.memmap(rank_dir / f"audio_tokens.{stem}.bin", dtype=np.int32, mode="r")
        text = np.memmap(rank_dir / f"text_tokens.{stem}.bin", dtype=np.int32, mode="r")
        table = pq.read_table(clips_path)
        rows = table.to_pylist()
        for row in rows:
            key = (str(row["source_id"]), int(row["clip_num"]))
            audio_start = int(row["audio_token_offset"]) // np.dtype(np.int32).itemsize
            audio_len = int(row["audio_token_length"])
            text_start = int(row["text_token_offset"]) // np.dtype(np.int32).itemsize
            text_len = int(row["text_token_length"])
            value = (
                tuple(int(x) for x in audio[audio_start: audio_start + audio_len]),
                tuple(int(x) for x in text[text_start: text_start + text_len]),
                row.get("clip_start"),
                row.get("clip_duration"),
            )
            if key in result:
                raise RuntimeError(f"duplicate interleave key: {key}")
            result[key] = value
    if not result:
        raise RuntimeError(f"no interleave clips found under {root}")
    return result


def _compare_interleave_cache(left: Path, right: Path) -> dict[str, int]:
    a = _collect_interleave_cache(left)
    b = _collect_interleave_cache(right)
    missing_left = set(b) - set(a)
    missing_right = set(a) - set(b)
    changed = {key for key in set(a) & set(b) if a[key] != b[key]}
    if missing_left or missing_right or changed:
        raise RuntimeError(
            "Interleave cache sets differ: "
            f"missing_from_left={len(missing_left)}, "
            f"missing_from_right={len(missing_right)}, "
            f"changed={len(changed)}"
        )
    return {
        "clips": len(a),
        "audio_tokens": sum(len(v[0]) for v in a.values()),
        "text_tokens": sum(len(v[1]) for v in a.values()),
    }


def _compare_megatron_outputs(left: Path, right: Path) -> dict[str, int]:
    structure_tokens = get_structure_tokens(TOKENIZER, required=["audio_start", "audio_end"])
    return compare_cutid_token_sets(
        left,
        right,
        trim_tolerance=TRIM_TOLERANCE,
        audio_start_id=int(structure_tokens["audio_start"]),
        audio_end_id=int(structure_tokens["audio_end"]),
    )


def _run_tokenize_case(case: TokenizeCase) -> dict[str, Any]:
    print(f"[1gpu] {case.name}", flush=True)
    one_root = ROOT / "tokenized" / f"{case.name}_1gpu"
    _run_pipeline(
        _tokenize_args(case, one_root),
        log_path=ROOT / "logs" / "tokenize" / f"{case.name}_1gpu.log",
        run_id=f"raw-canary-{RUN_STAMP}-{case.name}-1gpu",
    )
    one_out = _token_output_dir(case, one_root)
    one_stats = _stats_summary(one_out)

    print(f"[4gpu] {case.name}", flush=True)
    four_root = ROOT / "tokenized" / f"{case.name}_4gpu"
    _run_pipeline_multi(
        _tokenize_args(case, four_root),
        log_dir=ROOT / "logs" / "tokenize" / f"{case.name}_4gpu",
        world_size=4,
        run_id=f"raw-canary-{RUN_STAMP}-{case.name}-4gpu",
    )
    four_out = _token_output_dir(case, four_root)
    four_stats = _stats_summary(four_out)

    stat_mismatches = _compare_stats(one_stats, four_stats)
    if stat_mismatches:
        raise RuntimeError(f"{case.name} stats differ: {stat_mismatches}")

    if case.output_kind == "megatron":
        identity = _compare_megatron_outputs(one_out, four_out)
    else:
        identity = _compare_interleave_cache(one_out, four_out)

    materialize_result = None
    if case.materialize:
        print(f"[materialize] {case.name}", flush=True)
        materialize_root = ROOT / "materialized" / case.name
        materialize_args = [
            "run",
            "stage=materialize",
            "runtime.resume=false",
            *case.overrides,
            f"dataset.tokenization.input_shar_dir={case.shar_dir}",
            f"dataset.outputs.tokenized_dir={four_root}",
            f"dataset.outputs.name={case.output_name}",
            f"dataset.materialization.interleave.output_dir={materialize_root}",
        ]
        _run_pipeline(
            materialize_args,
            log_path=ROOT / "logs" / "materialize" / f"{case.name}.log",
            run_id=f"raw-canary-{RUN_STAMP}-{case.name}-materialize",
        )
        materialize_result = {"output_dir": str(materialize_root), "success": (materialize_root / "_SUCCESS").is_file()}
        if not materialize_result["success"]:
            raise RuntimeError(f"materialize success marker missing: {materialize_root}")

    return {
        "case": case.name,
        "kind": case.output_kind,
        "one_gpu_output": str(one_out),
        "four_gpu_output": str(four_out),
        "summary": four_stats,
        "identity": identity,
        "materialize": materialize_result,
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"root={ROOT}", flush=True)
    _cuda_preflight()

    rchiera_meta = _rchiera_metadata(ROOT / "metadata")

    samromur_shar = ROOT / "shar" / "samromur_wds_direct"
    rchiera_shar = ROOT / "shar" / "rchiera_wds_interleave"
    spc_shar = ROOT / "shar" / "spc_parquet"

    _convert_once(
        "samromur_wds_direct",
        _samromur_overrides(samromur_shar, ROOT / "convert_placeholder", "samromur_wds_direct"),
    )
    _convert_once(
        "rchiera_wds_interleave",
        _rchiera_overrides(rchiera_shar, ROOT / "convert_placeholder", "rchiera_wds_interleave", rchiera_meta),
    )
    _convert_once(
        "spc_parquet",
        _spc_overrides(spc_shar, ROOT / "convert_placeholder", "spc_parquet", mode="audio_text_interleaved"),
    )

    cases = [
        TokenizeCase(
            name="samromur_wds_direct",
            dataset="stage1/ups_all_lang_vad",
            output_name="samromur_wds_direct",
            output_kind="megatron",
            shar_dir=samromur_shar,
            overrides=tuple(_samromur_overrides(samromur_shar, ROOT / "unused", "samromur_wds_direct")),
        ),
        TokenizeCase(
            name="rchiera_wds_interleave",
            dataset="stage1/ups_all_lang_vad",
            output_name="rchiera_wds_interleave",
            output_kind="interleave",
            shar_dir=rchiera_shar,
            overrides=tuple(_rchiera_overrides(rchiera_shar, ROOT / "unused", "rchiera_wds_interleave", rchiera_meta)),
            materialize=True,
        ),
        TokenizeCase(
            name="spc_parquet_direct",
            dataset="cooldown/ccpodcasts",
            output_name="spc_parquet_direct",
            output_kind="megatron",
            shar_dir=spc_shar,
            overrides=tuple(_spc_overrides(spc_shar, ROOT / "unused", "spc_parquet_direct", mode="audio_text_direct")),
        ),
        TokenizeCase(
            name="spc_parquet_interleave",
            dataset="cooldown/ccpodcasts",
            output_name="spc_parquet_interleave",
            output_kind="interleave",
            shar_dir=spc_shar,
            overrides=tuple(_spc_overrides(spc_shar, ROOT / "unused", "spc_parquet_interleave", mode="audio_text_interleaved")),
            materialize=True,
        ),
    ]

    results = [_run_tokenize_case(case) for case in cases]
    payload = {"root": str(ROOT), "results": results}
    out = ROOT / "raw_audio_matrix_results.json"
    atomic_write_json(out, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    if "--preflight-only" in sys.argv:
        _cuda_preflight()
    else:
        main()
