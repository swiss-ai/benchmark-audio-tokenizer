# Audio Tokenization Pipeline — Audit & SFT Readiness (Post-Implementation)

**Branch:** `batch_tok` · **Date:** 2026-05-12 · **Supersedes:** `audio_tokenizer_audit_2026-05-09.md`

This audit replaces the May 9 audit. Since that document was written, SFT has landed end-to-end (schema, executor, audio token cache, three real dataset YAMLs, SLURM launcher), the stage CLI has been simplified, and a substantial number of the May 9 findings have been resolved. This document captures the **current** contract surface, marks what changed, and re-ranks the remaining work.

Findings are tagged **landed** (resolved since May 9), **carried** (still outstanding), or **new** (not in May 9). Action items are ranked by ROI at the end.

---

## 0. Headline state

- **501 tests passing** (was ~470 on May 9); 2 GPU-only deselected.
- **22.4K LOC** in `audio_tokenization/` production code (excluding tests); 11K tests.
- **Three real SFT datasets** configured and Hydra-plan-clean: TeleAntiFraud (216 audio hours, 33K conversations), Marco LongSpeech (8,655 audio hours, 204K conversations), AudioMCQ-GeminiCoT (speech subset).
- **Pipeline is production-ready for the SFT canary** modulo dev/lhotse PYTHONPATH; outstanding tasks are quality/efficiency tightening, not correctness gates.

---

## 1. Stage-boundary contracts (current)

### convert (prepare → SHAR)

Promises a complete, validated, resume-safe SHAR root. Boundary objects:

- `PrepareSpec` (Pydantic, frozen, `extra='forbid'`; `config/schema.py:142-300`). `fingerprint_payload()` is the resume source of truth.
- SHAR root: `worker_NN/_SUCCESS` (or `part-NNNNN/_SUCCESS` for `lhotse_recipe`) + root-level `shar_index.json`, `_SUCCESS`, `_PREPARE_STATE.json` (v2), `_worker_assignment.json`, `_shar_work_manifest.json`, `prepare_summary.json`.
- Per-cut invariants: `cut.id` immutable post-write; `cut.custom["interleave"]` = `{source_id, clip_num, clip_start, clip_duration}`; `cut.custom["rms_db"]`; optional `lang`, optional `text_tokens`. Cuts < MIN_RMS_DB dropped.
- **No standalone CLIs** — Hydra is the only entrypoint (commit `c38d3b0`, "Retire standalone prepare CLIs"). Family runners expose only `resolve / preflight / run`.

### tokenize (SHAR → audio tokens, GPU-distributed)

Promises a per-rank token cache plus a convoy-leader-published `_SUCCESS`. Boundary objects:

- `tokenize_state.json` (fingerprint of `TokenizeSpec` + resolved input dirs + upstream prepare provenance); `_SUCCESS` published only when **all** `rank_NNNN_stats.json` report `success=True`.
- `_shar_work_manifest.json` and `_tokenize_assignment.json` for rank assignment.
- Per-cut chunk outputs vary by `mode`:
  - `mode=audio_only`: Megatron `rank_XXXX_chunk_YYYY.{bin,idx}` + `cut_ids.jsonl.zst` sidecar; payload `[BOS, audio_start, audio_tokens..., audio_end, EOS]`.
  - `mode=audio_text + format=direct`: same Megatron triplet + cut-id sidecar; payload `[BOS, audio_start, audio_tokens..., audio_end, task_token_id, text_tokens..., EOS]`.
  - `mode=audio_text + format=interleaved`: structured v2 cache under `<partition>/rank_NNNN/clips.{stem}.parquet` + sibling `audio_tokens.{stem}.bin` / `text_tokens.{stem}.bin` + `_CACHE_LAYOUT.json`. Keyed by `(source_id, clip_num)`.
  - **`mode=audio_cache` (NEW since May 9)**: `rank_NNNN/audio_index.NNNNNN.parquet` + `audio_tokens.NNNNNN.bin` per-rank; manifest `_MANIFEST.json` at cache root with `tokenizer_fingerprint`, `token_dtype`, `audio_token_offset` semantics. Keyed by **`audio_id == cut.id`**. This is the asset-level cache that SFT materialize consumes.

Identity invariant: every emitted token region is keyed by `cut.id` (audio_only, audio_text-direct, audio_cache) or by `clip_id` column (interleaved parquet, where `clip_id == cut.id`).

### materialize (audio tokens → indexed dataset)

Two **product** outputs under one stage:

- `materialize.interleave` (unchanged): consumes the v2 structured cache, runs `interleave/shift_by_one.py`, emits Megatron `{prefix}.bin/.idx` (offsets + transcribe; stage2/lct buckets if `seq_threshold` set) + `metadata.json`.
- `materialize.sft` (**NEW since May 9**): consumes `audio_token_cache` + conversation parquets + text tokenizer; renders chat template, splices audio tokens at `<audio>` placeholders, emits per-worker Megatron shards. Cross-section validator gates: `sft.enabled` ⇒ `tokenize.mode == "audio_cache"` (or explicit `cache_dir`).

Both products share `stage="materialize"` and resume via `run_with_resume`. Schema validator enforces "at most one product enabled per dataset spec" today (the multi-product path is stubbed but blocked).

---

## 2. What landed since May 9 (the resolved findings)

### From the May 9 finding list

| ID | May 9 finding | Status today | Where |
|---|---|---|---|
| C-1 | Duplicated input resolution between stage adapter and runners | **landed** — convert input resolution lives in `prepare/runtime.py`, shared | post-audit cleanup |
| C-2 | Duplicated VAD/runtime preflight | **landed** — generic preflight in `prepare/runtime.py` | post-audit cleanup |
| C-3 | audio_dir/wds positional-tuple worker dispatch | **landed** — typed worker-args (`AudioDirWorkerArgs`, `WdsWorkerArgs`) | post-audit cleanup |
| C-4 | Three SHAR-index writers | **partial** — `runtime.build_shar_index_from_parts` is now the shared primitive; `lhotse_recipe` still has its own wrapper | `prepare/runtime.py:390+`, `prepare_lhotse_recipe_to_shar.py:253` |
| C-5 | `lhotse_recipe` rolls own pool + stats schema | **carried** — divergent | `prepare_lhotse_recipe_to_shar.py:386-421` |
| C-6 | Convert skip path runs twice | **carried** | `stages/convert.py` |
| C-7 | Per-family `_args_to_spec` repeated 5× | **landed** — standalone CLIs deleted; family runners no longer need `_args_to_spec` | commit `c38d3b0` |
| C-8 | `audio_dir` dataset-specific hot path | **deferred** — drop planned post-SFT (per `project_deferred_audio_dir_drop.md`) | `prepare_audio_dir_to_shar.py:66-407` |
| T-1 | `_invoke_pipeline_*` chain | **partial** — single-rank now writes assignment artifact + invokes assignment path directly | `stages/tokenize.py:163-189` |
| T-2 | Inlined dual-marker dance vs. shared `run_with_resume` | **carried** | `stages/tokenize.py:163-189` |
| T-3 | `audio_text_format` as if-branches in one handler | **carried** | `audio_text.py:101-322` |
| T-4 | Cross-format manifest fingerprint leakage | **carried** | `pipelines/lhotse/planning.py:166-178` |
| T-5 | Inert `audio_text_format`/`audio_text_task` for `audio_only` | **landed** — effective payload now conditional on mode (`stages/tokenize.py:_effective_tokenize_values`) | `stages/tokenize.py:86-99` |
| T-6 | `_distributed_rank_info` swallows arbitrary `import torch` exceptions | **carried** | `stages/tokenize.py:446+` |
| T-7 | Two near-duplicate Megatron writers (audio_only vs. audio_text-direct) | **carried** — and now `AudioCacheHandler` is a third near-duplicate; `_unsupervised_batch.py` extracted as a shared GPU→CPU helper | `audio_only.py`, `audio_text.py`, `pipelines/lhotse/audio_cache.py` |
| T-8 | Per-batch hot-loop instrumentation | **carried** | `pipelines/lhotse/core.py:469-566` |
| M-1 | Three near-duplicate Megatron-shard scaffolds (`shift_by_one`/`greedy`/`pattern`) | **carried + worsening** — `sft/materialize.py` is now a fourth scaffold (without parent merge); `MegatronShardedBuild` still missing | `interleave/{shift_by_one,greedy,pattern}.py`, `sft/materialize.py` |
| M-2 | Two parallel cache readers | **carried + worsening** — `token_cache.AudioTokenCache` is now a third reader keyed by `audio_id`; `cut_id_sidecar` for Megatron sidecars; `interleave/common.load_interleave_cache` for v2 structured cache | `token_cache.py`, `pipelines/cut_id_sidecar.py`, `interleave/common.py` |
| M-3 | SFT schema stub had no runnable executor | **landed** — `SftProductSpec`, `_resolve_sft_materialize_plan`, `sft/materialize.materialize_sft` all implemented with fork-pool, manifest validation, dup detection | `config/schema.py:681-715`, `stages/materialize.py:165-290`, `sft/materialize.py` |
| M-4 | `StageLabel` is a closed Literal missing `"sft"` | **landed (correctly)** — `StageLabel = Literal["convert", "tokenize", "materialize"]`; both interleave and sft callers pass `"materialize"`. Product names are no longer in the stage-label slot. | `stages/_resume.py:22` |
| M-5 | Pre-cleanup `SFT_STATE_FILE` unused | **landed** — `SFT_STATE_FILE = "products_sft_state.json"` now used by `_resolve_sft_materialize_plan` | `stages/materialize.py:33, 196` |
| M-6 | No per-token loss-mask in `IndexedDatasetBuilder` | **deferred (correctly)** — Megatron-LM handles loss masking at training time; pipeline does not emit masks. User explicitly chose this contract: "no attention_mask / no loss_mask / no labels". | `sft/materialize.py` (docstring), `docs/sft_audio_tokenization.md` |

### From the May 9 ranked action list

| # | May 9 item | Status |
|---|---|---|
| 1 | Define `cut_id_index.parquet` artifact at tokenize boundary | **superseded** — `audio_token_cache` index parquet plays this role for SFT; for audio_text-interleaved it's still the `(source_id, clip_num)`-keyed v2 structured cache |
| 2 | Per-token loss-mask sidecar | **deferred** (Megatron handles masking) |
| 3 | Extract `MegatronShardedBuild` helper | **carried + more urgent** — now 4 copies including SFT |
| 4 | Cross-section validator + `StageLabel="sft"` + `DocManifestSpec` | **partially landed** — `SftProductSpec` cross-section validator at `config/schema.py:751-756`; `StageLabel` corrected to use stage names (better than May 9 plan); `DocManifestSpec` not formalized but `SftProductSpec` covers messages/audio_ids column references |
| 5 | Implement `_run_sft_materialize` | **landed** — `_resolve_sft_materialize_plan` + `materialize_sft` |
| 6 | Hoist convert preflight + input resolution into `PrepareSpec` methods | **partial** — preflight + input resolution live in `prepare/runtime.py` but not on the spec directly |
| 7 | Typed worker-args for audio_dir + wds | **landed** |
| 8 | Hoist `lhotse_recipe` onto `run_pool_and_finalize` | **carried** |
| 9 | Split `audio_text_format` into handler subclasses | **carried** |
| 10 | Treat structured interleave cache as materialize-input format | **carried** (still the long-term optimum) |
| 11 | Collapse `_invoke_pipeline_*` chain | **carried** |
| 12 | Replace audio_dir → SHAR with clip-manifest Parquet | **deferred (correctly)** |

### Architectural changes not in May 9's plan (new contract surface)

| Change | Where | Why |
|---|---|---|
| **`stage: ???` (mandatory missing) + `optional stage: null`** | `configs/pipeline/config.yaml`, `stages/__init__.py:_require_single_stage` | Replaces `stage=all` magic. `run`/`clean` require exactly one stage. `plan`/`status` default to all-three. Reflects production reality: each stage has different cluster shape (CPU/IO, GPU-distributed, CPU) and cannot share a SLURM allocation. |
| **Drop `tokenization.enabled` chain in authoring** | `config/authoring.py` | Section build is now driven by output-field presence (`outputs.shar_dir` → convert section; `outputs.tokenized_dir` → tokenize section). One signal, not two. |
| **Drop `materialize_interleave: false` defaults from recipes** | `configs/pipeline/recipe/*.yaml` | Recipes no longer carry stale toggles. Eight recipe files cleaned. |
| **Resampling backend `"soxr"` promoted to tokenization section** | `configs/pipeline/recipe/*.yaml` | Was previously implicit; now explicit. Requires dev/lhotse PYTHONPATH (system lhotse 1.32.2 only has `["default", "sox"]`). |
| **`disabled_stage_plan.execute` raises `AssertionError`** | `stages/_plans.py` | Dead code under the explicit-stage contract. Future accidental callers now fail loud. |
| **`audio_token_cache` is a top-level module** | `audio_tokenization/token_cache.py` | Was originally under `sft/`; layering inversion (Lhotse pipeline depending on `sft/`) resolved by promoting cache to neutral home. Both `pipelines/lhotse/audio_cache.py` and `sft/materialize.py` import from it. |
| **`_unsupervised_batch.tokenize_unsupervised_batch` extracted** | `pipelines/lhotse/_unsupervised_batch.py` | Shared GPU→CPU strip+split kernel used by `audio_only` and `audio_cache` handlers. Eliminates one copy of the `[1:-1]` framing strip. |
| **`stages/materialize._resolve_product_cache_dir`** | `stages/materialize.py` | Unifies `_resolve_parquet_dir` (interleave) and `_resolve_sft_cache_dir` (sft) into one helper since both products' specs have `cache_dir: str | None`. |
| **`load_sft_chat_tokenizer` rename** | `sft/materialize.py` | Was originally `load_text_tokenizer`, same name as `prepare/text_ops.load_text_tokenizer` (different type contract). Renamed to disambiguate at call site. |

---

## 3. Drift findings — current state (new + carried)

Severity: **high** = correctness or production-blocker; **med** = quality/efficiency at scale; **low** = stylistic / micro-cleanup.

### convert (current)

| # | Finding | File:line | Sev | Tag |
|---|---|---|---|---|
| C-5 | **`lhotse_recipe` rolls its own pool + index + validate + summary** instead of using `run_pool_and_finalize`. Stats schema diverges (`num_cuts`/`total_duration`/`num_text_tokens` vs. `written`/`skipped`/`errors`). | `prepare_lhotse_recipe_to_shar.py:386-421` | med | carried |
| C-6 | **Convert skip path runs twice** — `run_convert` calls `try_skip_if_complete`; `_execute_convert_plan` calls it again. | `stages/convert.py:62-85, :254-278` | low | carried |
| C-8 | **`audio_dir` is a dataset-specific hot path.** Implemented + tested in a dropped worktree; resume planned post-SFT. | `prepare_audio_dir_to_shar.py:66-407` | med | deferred |

### tokenize (current)

| # | Finding | File:line | Sev | Tag |
|---|---|---|---|---|
| T-1 | **`_invoke_pipeline_*` chain.** Single-rank improvements landed (writes assignment artifact + invokes assignment path); the Russian-doll wrapper layers still exist. | `stages/tokenize.py:141-189, :192-275, :327-389, :606-630` | med | carried |
| T-2 | **Inlined dual-marker dance in tokenize vs. shared `run_with_resume` in materialize.** Two implementations of the same resume protocol. | `stages/tokenize.py:163-189` ↔ `stages/_resume.py:140-172` | med | carried |
| T-3 | **`audio_text_format` lives as if-branches in `AudioTextHandler`.** Tested in `setup_writer`, `process_batch`, `checkpoint_writer`, `finalize_writer`, `get_writer_state`. | `audio_text.py:101-322` | med | carried |
| T-4 | **Cross-format manifest fingerprint leakage.** `audio_only` manifests fingerprint differently on a SHAR with vs. without interleave columns even though audio_only doesn't use them. | `pipelines/lhotse/planning.py:166-178` | low | carried |
| T-6 | **`_distributed_rank_info` swallows arbitrary `Exception`.** ABI mismatches silently degrade to `local_rank=0`. | `stages/tokenize.py:446-452` | low | carried |
| T-7' | **Three near-duplicate handler writers.** `AudioOnlyHandler`, `AudioTextHandler._{direct,interleaved}_writers`, and now `AudioCacheHandler`. The `_unsupervised_batch.py` extraction helped with the GPU→CPU kernel but not the writer scaffolding. | `audio_only.py`, `audio_text.py`, `audio_cache.py` | med | carried + worsening |
| T-8 | **Per-batch hot-loop instrumentation.** W&B log call + tqdm postfix mutation + time snapshots per batch; rate-limit gate is *inside* `wandb_logger.log` after kwargs are evaluated. | `pipelines/lhotse/core.py:469-566` | low | carried |
| T-9 | **`stages/tokenize._preflight_sft_package_for_tokenize` couples tokenize to materialize.sft config.** Reads `spec.materialize.sft.{conversations_dir,messages_column,...}`. Documented as deliberate (fail-fast before GPU); the canonical preflight lives in materialize. | `stages/tokenize.py:141-169` | low | **new** |

### materialize (current)

| # | Finding | File:line | Sev | Tag |
|---|---|---|---|---|
| M-1 | **Four near-duplicate Megatron-shard scaffolds.** `interleave/{shift_by_one,greedy,pattern}.py` + `sft/materialize.py` each repeat fork-pool, partition loop, shard write. `MegatronShardedBuild` abstraction still missing — and SFT writes per-rank shards with **no parent merge** (vs. interleave's `_merge_shards`). | `interleave/{shift_by_one,greedy,pattern}.py`, `sft/materialize.py` | high | carried + worsening |
| M-2 | **Three parallel cache readers.** `interleave/common.load_interleave_cache` (v2 structured), `cut_id_sidecar` (Megatron sidecars), `token_cache.AudioTokenCache` (audio_id-keyed). The first two existed before; the third is new with SFT. None of them converse. | `interleave/common.py`, `pipelines/cut_id_sidecar.py`, `token_cache.py` | med | carried + worsening |
| M-7 | **No parent shard merge for SFT.** `sft/materialize.py` writes one `rank_NNNN_chunk_0000.{bin,idx}` per worker; never merged. interleave path merges via `_merge_shards`. Consumers see 64+ shards per dataset; inconsistent shape with interleave outputs. | `sft/materialize.py:103-114` | med | **new** (task #19) |
| M-8 | **Audio token cache hardcodes `int32` dtype** in 5 sites despite `_MANIFEST.json` documenting `token_dtype` as a field. Storage waste at scale (2 GB cache vs. 1 GB at uint16 for Marco), but more importantly a drift trap: any future dtype change requires coordinated touch of 5 sites. NOTE: pre-offset IDs in Apertus's combined 266K vocab can't actually fit in uint16; cache contract redesign required (task #12) — not a one-line `optimal_dtype` plumb. | `token_cache.py:53, 58, 72, 101, 152, 154, 159, 203, 357` | med | **new** |
| M-9 | **`audio_tokenizer_fingerprint` hashes only `audio_token_mapping.json`.** Changes to `tokenize.tokenizer.trim_last_tokens`, `sampling_rate`, or the WavTokenizer checkpoint silently don't invalidate the cache → silent corruption window for SFT consumers. | `token_cache.py:31-46` | high | **new** (task #18) |
| M-10 | **Text tokenizer loaded N times in workers** instead of once in parent (COW). At 64 workers × 200 MB HF AutoTokenizer ≈ 13 GB redundant RSS + 30-90s startup. The `_SHARED_AUDIO_CACHE` pattern exists right next to it. | `sft/materialize.py:_materialize_sft_worker` | med | **new** (task #17) |
| M-11 | **`SftMaterializeConfig` 1:1 duplicates `SftProductSpec`.** 10 fields × two parallel dataclasses. Every new SFT option requires three updates (schema + authoring + config). | `sft/materialize.py:29-41` ↔ `config/schema.py:681-695` | low | **new** (task #15) |
| M-12 | **`load_audio_token_cache` is per-row Python.** Builds 1M `AudioTokenSpan` frozen dataclasses + dict at load. ~5-10s startup + 250-500 MB RSS at 1M-row scale. Columnar Arrow + parallel ndarrays would be ~10× faster. | `token_cache.py:329-355` | med | **new** (task #8) |
| M-13 | **`assemble_sft_example` has redundant `np.asarray` wraps.** `cache.read()` already returns int32 ndarray; the outer `np.asarray(cache.read(...), dtype=np.int32)` is a no-op copy. **Partially landed** in this pass: outer wraps removed; `_tokenize_text` now returns ndarray. `_ensure_bos_eos` still allocates 3 small ndarrays per sample. | `sft/materialize.py:236-247, 378-385` | low | **new (partial fix landed)** |
| M-14 | **`max_seq_len` hard-cap loud-fails samples** above the threshold. Wrong primitive for SFT: per-sample length is a training-context decision, not a materialize gate. Should route to `stage2/`/`lct/` buckets via `seq_threshold` (`SftProductSpec.seq_threshold` is defined but not wired). | `sft/materialize.py:155-160` | med | **new** (task #16) |
| M-15 | **`disabled_stage_plan.execute` is unreachable.** Now raises `AssertionError`; landed in this audit's pass. | `stages/_plans.py:64-77` | resolved | landed |
| M-16 | **`_select_conversation_columns` was duplicated** in `sft/preflight.py` and `sft/materialize.py`. Now consolidated; preflight imports from materialize. | `sft/{preflight,materialize}.py` | resolved | landed |

### Cross-cutting

| # | Finding | File:line | Sev | Tag |
|---|---|---|---|---|
| X-1 | **Stale root dataset default.** Resolved by making the dataset group mandatory (`dataset: ???`) and requiring operators to choose a nested dataset such as `dataset=cooldown/infore2`. | `configs/pipeline/config.yaml:5` | resolved | landed |
| X-2 | **TokenizeMode mixes content axis with output-layer axis.** `audio_only / audio_text / audio_cache` — first two are *what* is tokenized, third is *how* the result is stored. Vokenizer factory at `vokenizers/wavtokenizer/__init__.py:26` already collapses `audio_only` + `audio_cache` to the same class, confirming the smell. | `config/schema.py:30` | low | **new** (accepted v1 simplification) |
| X-3 | **Flat `dataset/` directory with 28 YAMLs.** Resolved by nesting dataset specs under `dataset/{sft,cooldown,stage1,misc}/` while preserving internal `name` fields for artifact identity. | `configs/pipeline/dataset/` | resolved | landed |
| X-4 | **Missing group entries** for `tokenize/audio_cache.yaml` and `materialize/sft.yaml`. Accepted for now: SFT uses `recipe/parquet_sft_audio`; adding unused group entries would create extra config surface without simplifying operators' path. | `configs/pipeline/{tokenize,materialize}/` | low | accepted |

---

## 4. SFT readiness matrix (current)

| Surface | May 9 status | Today | Notes |
|---|---|---|---|
| `PrepareSpec` / SHAR layout | SFT-agnostic | unchanged | ✅ usable as-is |
| `cut.id` stability | preserved across modes | preserved + verified | ✅ usable as SFT join key |
| `audio_id == cut.id` join primitive | design proposal | **landed** | ✅ |
| `audio_token_cache` asset-level cache | design proposal | **landed**: `token_cache.py` (writer, reader, manifest, fingerprint validation) | ✅ tested with 19 SFT tests |
| Tokenize `audio_cache` mode | gap | **landed**: `mode=audio_cache` + `AudioCacheHandler` + cross-section validator (`tokenize.mode=audio_cache` required if SFT consumes derived cache) | ✅ |
| `SftProductSpec` schema | intentionally absent | **landed** | 13 fields covering examples/cache/tokenizer/output/runtime |
| `materialize.sft` executor | absent | **landed**: fork-pool with `_SHARED_AUDIO_CACHE` COW, per-worker shard write, manifest validation on cache load | ⚠️ no parent merge (M-7) |
| SFT preflight (catch misconfig pre-GPU) | not specified | **landed**: `sft/preflight.py` validates audio_id coverage, duration filter coverage, placeholder/audio_ids count match | ✅ |
| `IndexedDatasetBuilder` per-token loss mask | gap | **deliberately deferred** — pipeline does not emit masks; Megatron handles training-time masking | ✅ by design |
| Shared `MegatronShardedBuild` helper | missing (3 copies) | missing (**4 copies now** including SFT) | ⚠️ M-1 |
| `StageLabel` includes `"sft"` | one-line fix | **landed differently (correctly)** — `StageLabel = Literal["convert","tokenize","materialize"]`; product names retired from stage-label slot | ✅ |
| Cross-section validator for SFT | gap | **landed**: `config/schema.py:751-756` enforces `tokenize.mode == "audio_cache"` when SFT consumes derived cache | ✅ |
| Chat-template / placeholder convention | gap | **landed**: `<audio>` string placeholder (default), tokenizer's `apply_chat_template` renders messages; `<audio>` marks splice points; loud-fail on count mismatch | ✅ (string match; could harden to reserved token ID later) |
| `doc_manifest` schema | gap | **landed implicitly** via `SftProductSpec.{messages_column,audio_ids_column,examples_glob}` — schema is "conversation parquet has `sample_id`, `messages`, `audio_ids` columns" | ✅ for v1 |
| SLURM launcher | not specified | **landed**: `scripts/slurm/sft_audio.slurm` with documented per-stage invocations | ✅ |
| Real datasets configured | 0 | 3 (TeleAntiFraud, Marco LongSpeech, AudioMCQ-GeminiCoT) | ✅ |
| `pack_sft_dataset.py` offline utility | not specified | not implemented; data hand-authored offline | ⚠️ for ergonomics at scale (>5 datasets) |

**Bottom line:** SFT is **end-to-end production-ready for the canary** modulo the dev/lhotse PYTHONPATH dependency (which all SLURM scripts already set). Of the May 9 "**SFT**"-tagged work, items 4 and 5 are landed; items 2 (loss mask) and 6 (audio_dir) are deliberately deferred; item 3 (`MegatronShardedBuild`) is now more urgent because SFT became the fourth scaffold copy.

---

## 5. Ranked action items (current)

Severity reflects production-canary impact: P0 = blocks first SFT run; P1 = blocks scaling beyond ~3 datasets; P2 = quality/hygiene.

| # | Item | Files | Effort | Risk | Priority | Tag |
|---|---|---|---|---|---|---|
| 1 | **Expand `audio_tokenizer_fingerprint`** to cover `trim_last_tokens`, `sampling_rate`, WavTokenizer checkpoint (M-9) | `token_cache.py:31-46` | S | low | **P0** | silent-corruption gate; task #18 |
| 2 | **Replace `max_seq_len` hard-cap with `seq_threshold` routing** in SFT (M-14) | `config/schema.py:SftProductSpec`, `sft/materialize.py`, tests | S | low | **P1** | task #16 |
| 3 | **Add parent `_merge_shards` step to materialize_sft** (M-7) | `sft/materialize.py`, `interleave/common._merge_shards` | S | low | **P1** | task #19 |
| 4 | **Hoist text tokenizer to parent COW global** (M-10) | `sft/materialize.py` | XS | none | **P1** | task #17 |
| 5 | **Vectorize `load_audio_token_cache`** (M-12) | `token_cache.py:329-355` | S | low | **P1** | task #8 |
| 6 | **Eliminate `SftMaterializeConfig` 1:1 duplicate of `SftProductSpec`** (M-11) | `sft/materialize.py:29-41` | S | low | P2 | task #15 |
| 7 | **`np.asarray` no-op + `_ensure_bos_eos` allocation** drive-by (M-13) | `sft/materialize.py` | XS | none | P2 | task #9 (partial done) |
| 9 | **Within-chunk dup-check test for `AudioTokenCacheWriter`** | `tests/test_sft_materialization.py` | XS | none | P2 | task #14 |
| 10 | **Decide: audio cache dtype contract** (M-8) — keep `int32` (pre-offset IDs in 266K combined vocab can't fit uint16) or migrate to raw codes + sentinels (50% storage savings, contract redesign) | `token_cache.py`, manifest schema | S–M | med (cache rebuild) | P2 (deferred) | task #12 |
| 11 | **Extract `MegatronShardedBuild` helper** parameterized by `(emit_fn, route_keys, dtype, num_workers, tmp_dir)`; replaces 4 copies (shift_by_one + greedy + pattern + sft) | `interleave/common.py`, `interleave/*.py`, `sft/materialize.py` | M | low–med | P2 | M-1; carried from May 9 #3 |
| 12 | **Extract `ChunkWriterBase`** (atomic tmp-write + fsync + parquet sidecar + `atomic_replace_files`); replaces 3 copies (`AudioTokenCacheWriter`, `StructuredCacheChunkWriter._PartitionShardWriter`, `open_chunk_writer`) | `pipelines/shard_io.py` | M | low | P2 | **new** |
| 13 | **Promote `audio_text_format` to two handler subclasses + extract third writer to match** (T-3, T-7') | `audio_only.py`, `audio_text.py`, `audio_cache.py`, `core.py` | M | med | P2 | carried + worsened by audio_cache addition |
| 14 | **Collapse `_invoke_pipeline_*` chain** (T-1, T-2) | `stages/tokenize.py` | S | low | P2 | carried |
| 15 | **Move audio-cache constants** (`_INDEX_SCHEMA`, `MANIFEST_FILENAME`, `AUDIO_TOKEN_CACHE_FORMAT`) **to `contracts/artifacts.py`** | `token_cache.py`, `contracts/artifacts.py` | XS | none | P2 | **new** |
| 16 | **Optional dry-run mode for `materialize.sft`** (capacity planning / oversize detection) | `sft/materialize.py` | M | none | P3 (optional) | task #11 |
| 17 | **Treat structured interleave cache as materialize-input format, not tokenize-output mode** | `pipelines/lhotse/audio_text.py`, `output_layout.py`, `interleave/shift_by_one.py` | L | high (cache migration) | P3 (long-term optimum) | carried (May 9 #10) |
| 18 | **Drop `interleave/greedy.py` + `interleave/pattern.py` if no operator use** | `interleave/{greedy,pattern}.py` + tests | S (deletion) | low (after confirmation) | P3 | **new** — 1,069 + 521 LOC + pattern.py + tests; CLI-only, not stage-graph-wired |
| 19 | **Config tree reorg**: nest `dataset/` under `{sft,cooldown,stage1,misc}/`; add `tokenize/audio_cache.yaml` + `materialize/sft.yaml` group entries; add `README.md` (X-3, X-4) | `configs/pipeline/` | S | none | P3 | **new** — defer until post-canary |
| 20 | **`audio_dir` family drop** | `prepare/prepare_audio_dir_to_shar.py`, `config/schema.py`, `config/authoring.py` | M | low | P3 | deferred per memory `project_deferred_audio_dir_drop.md` |

### Recommended execution order

**Pre-canary (P0 + selected P1):**
1. #1 fix stale `dataset` default — XS, zero risk
2. #2 expand `audio_tokenizer_fingerprint` — closes silent-corruption window before any production run
3. #5 hoist text tokenizer to parent (M-10) — saves 30-90s startup × N workers; cheap
4. #3 `seq_threshold` routing (replaces `max_seq_len` hard-cap) — design lock-in, matches operator mental model

**Marco canary** — once 1-4 land, run the canary. Validates the contract end-to-end on real data.

**Post-canary (P1):**
5. #4 add parent `_merge_shards` to SFT
6. #6 vectorize `load_audio_token_cache`
7. #10 audio cache dtype contract decision

**Quality (P2):**
8. #7, #8, #9, #12, #15
9. #11 + #13 — `MegatronShardedBuild` and the handler-subclass split (these are the carried architectural moves)
10. #14 collapse `_invoke_pipeline_*` chain

**Future (P3):**
- #16 dry-run, #17 cache-format unification, #18 greedy/pattern drop, #19 config reorg, #20 audio_dir drop

---

## 6. Architectural state — current global view

| Axis | State | Comment |
|---|---|---|
| Stage decomposition (convert/tokenize/materialize) | ✅ load-bearing, contracts honored | Single-stage-per-`run` enforces cluster shape |
| Family-runner ingestion (5 prepare families) | ✅ unified `resolve/preflight/run` triple | Standalone CLIs retired |
| Resume protocol (dual marker: `_SUCCESS` + state JSON) | ✅ shared via `run_with_resume` for convert + materialize; tokenize inlines its own (T-2) | Drift detection works |
| Fingerprint hygiene (audio tokenize cache invalidation) | ⚠️ M-9 — covers only `audio_token_mapping.json`; missing tokenizer config knobs | P0 to fix before production |
| Asset-level audio cache (asset-keyed reusable tokens) | ✅ `audio_token_cache` + manifest + tokenizer-identity check + dup detection | New since May 9 |
| Materialize products (interleave + sft) | ✅ both wired; cross-section validator enforces compatibility | Schema validator forbids both enabled in one spec |
| Shard scaffold duplication | ⚠️ M-1 — 4 copies including SFT; `MegatronShardedBuild` still missing | P2 |
| Cache reader duplication | ⚠️ M-2 — 3 parallel readers; none converse | P2 |
| Hot-path efficiency (audio tokenize) | ✅ multi-rank GPU, batched inference, vectorized post-processing, `optimal_dtype` for Megatron output | Untouched in this round |
| Hot-path efficiency (materialize) | ⚠️ shift_by_one is at-bar (fork-pool + merge); sft is below-bar (per-worker shards, no merge, no COW tokenizer) | Tasks #4, #5, #6, #19 |
| Test rigor | ✅ 501 passing; 51% test-to-code ratio | SFT covered with 19 dedicated tests |
| Documentation | ✅ `docs/sft_audio_tokenization.md` + this audit | Stage1 reconversion todo doc still current |

---

## 7. Assumptions

- No backward compatibility for v1 prepare states (per `feedback_prepare_state_v2`).
- `cut.id` stability is preserved across all tokenize modes including the new `audio_cache` (verified).
- Apertus's combined audio+text vocab (~266K) forces `int32` token storage in both the cache and the final Megatron bin/idx; uint16 would require a cache-contract redesign (raw codes + sentinels).
- `materialize.interleave` stays first-class; `materialize.sft` added alongside, schema validator enforces at-most-one-enabled.
- SFT messages format: chat-template-renderable, single canonical `<audio>` placeholder; tokenize text once per span on render.
- Megatron-LM handles training-time loss masking; pipeline emits no masks (deliberate, captured in `docs/sft_audio_tokenization.md`).
- Text tokenizer (HF AutoTokenizer with chat template) is corpus-iteration-stable; not iterated per chat-template variant. (If this changes, hoist to a separate group config under `training_target/`.)
- `dev/lhotse` PYTHONPATH is set by SLURM scripts (memory: `feedback_use_dev_lhotse`).

---

## 8. Net change since May 9

| Dimension | May 9 | Today | Delta |
|---|---|---|---|
| Tests passing | ~470 | 501 | +31 |
| Production LOC | ~21K | ~22.4K | +1.4K (SFT + audio_token_cache + preflight) |
| SFT datasets configured | 0 | 3 | +3 |
| Stage CLI surface | `stage=all` + chained `tokenization.enabled` | mandatory `stage=???` + presence-driven authoring | retired magic |
| `materialize.sft` | absent (intentional) | live + tested | landed |
| `audio_token_cache` artifact | absent | full surface (writer + reader + manifest + fingerprint validation + dup detection) | new |
| Open P0 items | 0 (no production target) | 2 (#1, #2) | new gate |
| Open architectural debts (M-1, M-2) | 2 | 2 (worsened — both grew a 3rd/4th copy) | unchanged count, increased pressure |

The May 9 audit's recommended order — **#2 → #3 → #4 → #1 → #5** — landed as: #5 (executor) first via the SFT subpackage, #4 (validator) alongside, #1 (cache artifact) via `audio_token_cache`, with #2 (loss mask) deliberately retired and #3 (`MegatronShardedBuild`) deferred. The deferral on #3 is the biggest carried debt and is now the highest-leverage architectural cleanup pending.

---

## 9. Recommended next single action

**#1 in one focused commit**:

1. `token_cache.py:audio_tokenizer_fingerprint`: extend to hash `(tokenizer.sampling_rate, tokenizer.trim_last_tokens, tokenizer_checkpoint_sha256)` alongside the mapping file; bump `AUDIO_TOKEN_CACHE_FORMAT` to `v2`; reject `v1` caches with a clear migration message.

The config-layout gate is now closed; the cache fingerprint hardening is the remaining P0 correctness gate.
