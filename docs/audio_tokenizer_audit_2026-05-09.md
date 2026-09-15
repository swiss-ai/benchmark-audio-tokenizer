# Audio Tokenization Pipeline — Audit & SFT Readiness

**Branch:** `batch_tok` &nbsp;·&nbsp; **Date:** 2026-05-09 &nbsp;·&nbsp; **Scope:** prepare → tokenize → materialize, with audio-SFT support as the forward-looking constraint.

This audit is contract-first: each section starts from the boundary contract (what should be true at the stage edge), then names where the current code drifts from it and where SFT will collide. Findings are ranked at the end with effort / risk / SFT-required vs. cleanup.

SFT shape used throughout: multi-turn conversations whose user turns are audio (spoken or uploaded file with instructions) **or** text, and whose assistant turns are text. Loss masked to assistant turns. Tokenize is upstream; SFT lives in `materialize`.

### Post-audit cleanup status

Applied after this audit:

- C-1/C-2: convert input resolution and generic prepare preflight now live in `prepare/runtime.py` and are shared by the convert stage and prepare runners.
- C-3: `audio_dir` and `wds` conversion workers now use typed worker-args (`AudioDirWorkerArgs`, `WdsWorkerArgs`) instead of positional tuples.
- T-1: the single-rank tokenize path now writes the same assignment artifact as distributed tokenize and invokes the assignment path directly.
- Legacy prepare-state provenance fallback was removed. If an upstream SHAR root has `_PREPARE_STATE.json`, it must be current typed/versioned state; old stage-2 SHARs with unversioned state are intentionally no longer supported for tokenize.
- The active SFT config/schema/executor stub was removed. SFT remains a design target in this audit, but `materialize.sft` should not be reintroduced until the assembler and tests land with it.
- Future cleanup #12 was added: replace the VoxPopuli-style `audio_dir -> SHAR` hot path with normalized clip-manifest Parquet plus a generic path/offset materializer.

---

## 1. Stage-boundary contracts

### convert (prepare → SHAR)

Promises a complete, validated, resume-safe SHAR root. Boundary objects:

- `PrepareSpec` (typed Pydantic, frozen, `extra='forbid'`; `config/schema.py:350-385`). `fingerprint_payload()` is the resume source of truth.
- SHAR root: `worker_XX/_SUCCESS` (or `part-NNNNN/_SUCCESS` for `lhotse_recipe`) + root-level `shar_index.json`, `_SUCCESS`, `_PREPARE_STATE.json` (versioned, current=2), `_worker_assignment.json`, `_shar_work_manifest.json` (the durable handoff to tokenize), `prepare_summary.json`.
- Per-cut invariants: `cut.id` immutable post-write; `cut.custom["interleave"]` = `{source_id, clip_num, clip_start, clip_duration}`; `cut.custom["rms_db"]`; optional `lang`, optional `text_tokens`. Cuts < `MIN_RMS_DB` (-50 dB) dropped.
- `validate_shar_directory` runs before root `_SUCCESS`.

### tokenize (SHAR → audio tokens, GPU)

Promises a per-rank Megatron-or-structured-cache token cache plus a convoy-leader-published `_SUCCESS`. Boundary objects:

- `tokenize_state.json` (fingerprint of `TokenizeSpec` + resolved input dirs + upstream prepare provenance); `_SUCCESS` published only when **all** `rank_NNNN_stats.json` report `success=True` (`stats_reducer.py:66-95`).
- `_shar_work_manifest.json` and `_tokenize_assignment.json` for rank assignment.
- Per-cut chunk outputs vary by `audio_text_format`:
  - `mode=audio_only`: Megatron `rank_XXXX_chunk_YYYY.{bin,idx}` + `cut_ids.jsonl.zst` sidecar; payload `[BOS, audio_start, audio_tokens..., audio_end, EOS]`.
  - `mode=audio_text + format=direct`: same Megatron triplet + cut-id sidecar; payload `[BOS, audio_start, audio_tokens..., audio_end, task_token_id, text_tokens..., EOS]`.
  - `mode=audio_text + format=interleaved`: structured v2 cache under `<partition>/rank_NNNN/clips.{stem}.parquet` + sibling `audio_tokens.{stem}.bin` / `text_tokens.{stem}.bin` + `_CACHE_LAYOUT.json`. **Keyed by `(source_id, clip_num)`, not by `cut.id`.**
- Identity invariant: every emitted token region is keyed by `cut.id` (Megatron sidecar) or by `clip_id` column (interleaved parquet, where `clip_id == cut.id`).

### materialize (audio tokens -> indexed dataset)

Active product:

- `materialize.interleave` consumes the v2 structured cache, runs `interleave/shift_by_one.py`, emits Megatron `{prefix}.bin/.idx` outputs (`offset_0`, `offset_1`, `transcribe`; under `stage2/` and `lct/` if `seq_threshold` set) + `metadata.json`. Resume via `run_with_resume`; fingerprint embeds upstream `tokenize_state.json` verbatim.

SFT is intentionally design-only after post-audit cleanup: the previous `materialize.sft` schema/config/executor stub was removed because it advertised a product that could only raise `NotImplementedError`. Reintroduce it with the assembler, doc-manifest schema, token-cache join contract, and tests in the same change.

---

## 2. Drift findings (where code violates its own contract)

### convert

| # | Finding | File:line | Sev |
|---|---|---|---|
| C-1 | **Duplicated input resolution.** `_resolve_convert_inputs` globs every family's inputs; each runner re-globs the same paths. Two sources of truth that can disagree if a glob expands differently between calls. | `stages/convert.py:103-160` ↔ `prepare_*_to_shar.py` (5 files) | high |
| C-2 | **Duplicated VAD/runtime preflight.** `_preflight_convert_plan` calls `validate_prepare_runtime` + `_validate_vad_thresholds`; each runner *also* calls `validate_prepare_runtime` and inlines the VAD threshold check char-for-char. | `stages/convert.py:178-251`, `prepare_wds_to_shar.py:603-612`, `prepare_audio_dir_to_shar.py:360-369` | high |
| C-3 | **`audio_dir` and `wds` workers bypass the typed worker spec.** 15/16-tuple positional dispatch on `audio_dir`, 20-tuple on `wds`. parquet/HF migrated to `ColumnarWorkerArgs`; VAD families did not. | `prepare_audio_dir_to_shar.py:74-117`, `prepare_wds_to_shar.py:352-373` | high |
| C-4 | **Three SHAR-index writers** wrap the same `build_shar_index_from_parts`; `lhotse_recipe.build_shar_index` shadows `runtime.build_shar_index` with a different signature. | `prepare/runtime.py:390-417`, `prepare_lhotse_recipe_to_shar.py:253-261` | med |
| C-5 | **`lhotse_recipe` rolls its own pool + index + validate + summary** instead of using `run_pool_and_finalize`. Different stats schema (`num_cuts`/`total_duration`/`num_text_tokens`) than the shared aggregator's (`written`/`skipped`/`errors`). | `prepare_lhotse_recipe_to_shar.py:386-421` | med |
| C-6 | **Convert skip path runs twice.** `run_convert` calls `try_skip_if_complete`; `_execute_convert_plan` calls it again. Defensive, but the second only fires after the runner is resolved. | `stages/convert.py:62-85`, `:254-278` | low |
| C-7 | **Per-family `_args_to_spec` flat→nested mapping repeated 5×.** Configuration-as-data living in runtime; every schema field add requires 5 mirror updates. Standalone CLIs only. | `prepare_*_to_shar.py:_args_to_spec` (5 files) | low |
| C-8 | **`audio_dir` is a dataset-specific hot path.** Used by VoxPopuli VAD, it mixes recursive audio indexing, VAD JSONL parsing, VAD packing, audio IO, and SHAR writing in one worker loop. Long term, normalize `audio_dir + VAD JSONL` into a clip-manifest Parquet and reuse a generic path/offset materializer. | `prepare_audio_dir_to_shar.py:66-407`, `configs/pipeline/dataset/stage1/voxpopuli_vad.yaml:2-31` | med |

### tokenize

| # | Finding | File:line | Sev |
|---|---|---|---|
| T-1 | **`_invoke_pipeline_*` chain (parameter sprawl).** Russian-doll: `_execute_tokenize_plan` → `_invoke_pipeline_for_rank` → `_invoke_pipeline_for_assignment` → `_invoke_pipeline`. Single-rank wrapper duplicates work the distributed path also does. | `stages/tokenize.py:141-189`, `:192-275`, `:327-389`, `:606-630` | med |
| T-2 | **Inlined dual-marker dance vs. shared `run_with_resume`.** Materialize uses `run_with_resume`; tokenize inlines `prepare_output_for_work` + relies on `maybe_publish_terminal_artifacts` for `_SUCCESS` (convoy-leader race). Two implementations of the same protocol. | `stages/tokenize.py:163-189` ↔ `stages/_resume.py:140-172` | med |
| T-3 | **`audio_text_format` lives as if-branches in one handler.** `AudioTextHandler` re-tests format in every method (`setup_writer`, `process_batch`, `checkpoint_writer`, `finalize_writer`, `get_writer_state`). Two handler subclasses would express it cleanly. | `audio_text.py:101-322` | med |
| T-4 | **Cross-format manifest fingerprint leakage.** `SharWorkManifest.fingerprint` always includes interleave-only metadata coverage counts, so audio_only manifests fingerprint differently on a SHAR with vs. without interleave columns even though audio_only doesn't use them. | `pipelines/lhotse/planning.py:166-178` | low |
| T-5 | **Inert `audio_text_format` / `audio_text_task` for `audio_only`.** Configs carry `audio_text_format: direct, audio_text_task: transcribe` lines that are silently ignored by the handler but flow into the fingerprint and output dir. Inert noise that bloats the resume key. | `configs/pipeline/tokenize/audio_only.yaml`, `config/schema.py:511-540` | low |
| T-6 | **`_distributed_rank_info` swallows arbitrary `Exception` from `import torch`.** Any torch import error (CUDA absent, ABI mismatch — see `feedback_torchaudio_fix`) silently degrades to `local_rank=0`. | `stages/tokenize.py:446-452` | low |
| T-7 | **Two near-duplicate Megatron writers.** `AudioOnlyHandler` and `AudioTextHandler._{setup,checkpoint,finalize}_writer_direct` repeat the same 7-tuple `open_chunk_writer` unpack and finalize block. Pattern repeats; abstraction is missing. | `audio_only.py:36-134` ↔ `audio_text.py:107-322` | med |
| T-8 | **Per-batch hot-loop instrumentation.** W&B log call + tqdm postfix mutation + time snapshots per batch; rate-limit gate is *inside* `wandb_logger.log` after kwargs are evaluated. | `pipelines/lhotse/core.py:469-566` | low |

### materialize

| # | Finding | File:line | Sev |
|---|---|---|---|
| M-1 | **Three near-duplicate Megatron-shard scaffolds.** `interleave/{shift_by_one,greedy,pattern}.py` each repeat tmp-dir lifecycle, partition loop, fork pool, merge keys, dtype selection, metadata.json. Pattern repeats 3+ times → shared `MegatronShardedBuild` abstraction missing. | `interleave/shift_by_one.py:560-660`, `interleave/greedy.py`, `interleave/pattern.py` | high |
| M-2 | **Two parallel cache readers** that never converse: `interleave/common.py:218-304` (v2 structured cache, audio+text aligned, run-detected) vs. `cut_id_sidecar.py:62-117` (per-chunk Megatron with cut-id-by-line). SFT will need the latter; interleave the former. | `interleave/common.py`, `pipelines/cut_id_sidecar.py` | med |
| M-3 | **Pre-cleanup SFT schema stub had no runnable executor.** Resolved post-audit by removing the active `materialize.sft` schema/config path until the real assembler lands. | `config/schema.py`, `stages/materialize.py` | med |
| M-4 | **`StageLabel` is a closed Literal** that doesn't include `"sft"`. Adding SFT later requires updating `stages/_resume.py:22`. | `stages/_resume.py:22` | low |
| M-5 | **Pre-cleanup `SFT_STATE_FILE` constant was unused.** Resolved post-audit by removing the constant and `_run_sft_materialize` stub. | `stages/materialize.py` | low |
| M-6 | **No per-token loss-mask in `IndexedDatasetBuilder`.** Megatron `sequence_modes` is per-sequence and isn't a substitute. SFT *requires* per-token loss masks for assistant-only loss. **This is foundational and missing.** | `utils/indexed_dataset/indexed_dataset_megatron.py` | high |

---

## 3. SFT collision / readiness matrix

| Surface | Status today | Action needed |
|---|---|---|
| `PrepareSpec` / SHAR layout | SFT-agnostic; downstream of convert | none — convert needs no SFT changes |
| `cut.id` stability | immutable, propagates through all tokenize modes (sidecar or `clip_id` column) | none — usable as SFT join key |
| `tokenize.mode=audio_only` "emit tokens, no transcript required" | works today; `data.py:152-160` only filters for text when `mode==audio_text` | **none** — already SFT-ready for audio-only user turns |
| `tokenize.mode=audio_text` cut-id index | structured cache keyed by `(source_id, clip_num)`, not `cut.id` | **gap** — SFT needs a `cut_id → (chunk, doc, byte_range)` artifact for audio-text mode if SFT consumes audio-text outputs |
| `cut_id_index.parquet` first-class artifact | only sidecars exist; the join index is reconstructable but not published | **gap** — define and emit at tokenize boundary |
| `audio_text_format` axis | direct / interleaved live as if-branches in one handler | not collision per se, but SFT should *not* become a third format — SFT is materialize-layer |
| `SftProductSpec` schema | intentionally absent after cleanup | add with the SFT assembler, not before |
| `_run_sft_materialize` executor | intentionally absent after cleanup | **gap** — implement with schema/config/tests |
| `IndexedDatasetBuilder` per-token loss mask | absent | **gap** — foundational; assistant-only loss requires it |
| Shared `MegatronShardedBuild` helper | missing (3 copies in interleave/) | **gap** — without it, SFT becomes the 4th copy |
| `StageLabel` includes `"sft"` | no | one-line fix |
| Cross-section validator (`sft.enabled` ⇒ upstream tokenize mode/cache contract) | absent | **gap** — would catch null-paths config at validate time |
| Chat-template / placeholder convention | undefined | **gap** — SFT design decision (e.g., `<|audio:cut_id|>` sentinel; resolved by Jinja then substituted) |
| `doc_manifest` schema | unspecified field in yaml; no Pydantic backing | **gap** — define `DocManifestSpec`: required columns, role enum, content-type enum |

**Bottom line:** convert is SFT-ready as-is. Tokenize is *almost* SFT-ready — `audio_only` works, but a stable join index is missing. Materialize needs three foundational pieces (loss-mask, shared scaffold, cross-section validator) plus the actual SFT executor. None of this requires invasive changes to load-bearing code; everything is additive plus one Pydantic validator.

---

## 4. Ranked simplifications

Items marked **SFT** are required to land SFT. Others are cleanup.

| # | Item | Files | Effort | Risk | Tag |
|---|---|---|---|---|---|
| 1 | **Define `cut_id_index.parquet` artifact** at tokenize boundary (cols: `cut_id, chunk_prefix, doc_index, num_audio_tokens`). Emit from `maybe_publish_terminal_artifacts` for both audio_only and audio_text modes by reading sidecars / parquet schemas already on disk. | `pipelines/lhotse/stats_reducer.py`, new `contracts/sft_cache.py` | S | low | **SFT** |
| 2 | **Add per-token loss-mask sidecar** to `IndexedDatasetBuilder` (option A: extend `add_item(..., loss_mask=...)` + write `.mask.bin`; option B: parallel Megatron dataset with the mask as token data). Touches the writer all products depend on; needs golden-test protection. | `utils/indexed_dataset/indexed_dataset_megatron.py`, `interleave/common._merge_shards` | M | med | **SFT** |
| 3 | **Extract `MegatronShardedBuild` helper** parameterized by `(emit_fn, route_keys, dtype, num_workers, tmp_dir)`. Replaces three copies in `interleave/{shift_by_one,greedy,pattern}.py`; SFT becomes a fourth caller, not a fourth copy. | `interleave/common.py`, `interleave/{shift_by_one,greedy,pattern}.py` | M | low | **SFT** (per "rule of three" — landing SFT first means doing the abstraction *after* a fourth duplicate exists) |
| 4 | **Cross-section validator** in `DatasetSpec._validate_cross_section_invariants`: `materialize.sft.enabled and audio_token_cache_dir is None` ⇒ require `tokenize.mode='audio_only'`. Add `StageLabel="sft"` to `_resume.py:22`. Define `DocManifestSpec` schema (parquet/jsonl flag, required columns, role enum). | `config/schema.py:614-642`, `stages/_resume.py:22` | S | low | **SFT** |
| 5 | **Implement `_run_sft_materialize`** wiring: `run_with_resume(stage_label="sft", state_filename=SFT_STATE_FILE, ...)`; new `audio_tokenization/sft_assembler/main.py` that joins `doc_manifest` × `cut_id_index` via cut-id, renders chat template with audio sentinels, substitutes audio tokens, builds Megatron output via the `MegatronShardedBuild` helper from #3. | `stages/materialize.py:110-118`, new `sft_assembler/` module | M | low–med | **SFT** (actual feature work) |
| 6 | **Hoist convert preflight + input resolution into `PrepareSpec` methods** (`spec.resolved_inputs()`, `spec.preflight()`); delete duplicates inside each runner. Removes C-1 + C-2 in one shape change. | `stages/convert.py`, all 5 `prepare_*_to_shar.py` | M | low | cleanup |
| 7 | **Promote `audio_dir` + `wds` workers to typed worker-args** mirroring `ColumnarWorkerArgs`. Kill 15/16/20-tuple dispatch. | `prepare_audio_dir_to_shar.py`, `prepare_wds_to_shar.py`, `columnar.py` | S | low | cleanup |
| 8 | **Hoist `lhotse_recipe` onto `run_pool_and_finalize`**; unify stats schema with the shared aggregator. | `prepare_lhotse_recipe_to_shar.py`, `prepare/runtime.py` | M | med | cleanup |
| 9 | **Promote `audio_text_format` to two handler subclasses** (`AudioTextDirectHandler`, `InterleavedCacheHandler`) sharing a writer protocol with `AudioOnlyHandler`. Resolves T-3, T-7. | `pipelines/lhotse/audio_only.py`, `audio_text.py`, `core.py:671-678` | M | med | cleanup; pre-condition for #10 |
| 10 | **Treat the structured interleave cache as a materialize-input format, not a tokenize-output mode.** Tokenize emits one shape: `(cut_id, audio_token_bytes, text_token_bytes?, supervision_metadata)` + `cut_id_index.parquet`. The current "interleaved" partitioned cache becomes a materialize-input view, generated by a thin reshaper. Both `interleave` and `sft` then consume the same upstream artifact. | `pipelines/lhotse/audio_text.py`, `pipelines/shard_io.py`, `output_layout.py`, `interleave/shift_by_one.py` | L | high (cache-format migration) | cleanup; long-term global optimum |
| 11 | **Collapse `_invoke_pipeline_*` chain** into one planner + one execute. | `stages/tokenize.py` | S | low | cleanup |
| 12 | **Replace `audio_dir → SHAR` with normalized clip-manifest Parquet.** For VoxPopuli VAD, first write one row per final packed clip (`sample_id`, `source_id`, `clip_num`, `audio_path`, `clip_start_sec`, `clip_duration_sec`, `language`, metadata), then reuse a generic Parquet materializer. Optional later phase: pack source audio into few large archive shards to reduce raw-audio inode pressure. | new normalizer + materializer, `prepare_audio_dir_to_shar.py`, `recipe/audio_dir_audio_only.yaml` | M-L | med | future cleanup |

---

## 5. Recommended order if SFT is the goal

The contract-first order is **#2 → #3 → #4 → #1 → #5** (all SFT-tagged items). Rationale:

- **#2 (loss-mask sidecar) is the only foundational writer change.** Doing it first means SFT lands on a writer that already supports masks, instead of a mid-implementation contract bump.
- **#3 (`MegatronShardedBuild`)** must precede #5 or SFT becomes a fourth scaffold copy by construction. Doing it before SFT also gives the existing three interleave strategies the simplification benefit retroactively.
- **#4 (validator + StageLabel + DocManifestSpec)** is a one-PR contract pass that fails-loud at config time on incompatible `sft.enabled` configurations. Cheap; lands the "what does an SFT config look like" question.
- **#1 (`cut_id_index.parquet`)** is the artifact that joins SFT's `doc_manifest.audio_ref` to upstream tokens. Until this exists, SFT either reconstructs from sidecars (slow + redundant) or is locked to `mode=audio_only`.
- **#5 (executor)** is then mechanical: the boundary contracts above already exist, so the assembler is "join, render template, substitute audio tokens, mask, write."

Cleanup items #6 – #12 are independent and can land in any order. **#10 is the global optimum** but explicitly flagged as L-effort with cache-format migration risk; it should not block SFT — SFT can land on the current dual-mode tokenize and benefit from #10 later. **#12 is specifically future cleanup:** the current VoxPopuli `audio_dir` path works, but new SFT code should not build on that special hot path.

### Assumptions

- No backward compatibility for v1 prepare states (per project memory; V=2 is current).
- `cut.id` stability is preserved across all tokenize modes (verified).
- SFT manifest format will be parquet with explicit `conversation_id`, `turns` schema (see #4 — `DocManifestSpec`); jsonl is a convenience subset, not the design point.
- Assistant turns are text-only at first; assistant-audio is a future axis that doesn't affect the current contract.
- `materialize.interleave` stays first-class; SFT is added alongside, not replacing it.
