from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from audio_tokenization.utils.indexed_dataset.cut_id_sidecar import collect_cutid_token_pairs


class _Tokenized:
    def __init__(self, input_ids: list[int]):
        self.input_ids = input_ids


class _DummyTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self) -> int:
        return 1000

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        assert tokenize is False
        assert add_generation_prompt is False
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return _Tokenized([10 + (ord(ch) % 50) for ch in text])


def _write_tokenizer_mapping(path: Path, *, audio_start: int = 11, audio_end: int = 12) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "audio_token_mapping.json").write_text(
        json.dumps({
            "audio_token_offset": 100,
            "structure_tokens": {
                "audio_start": audio_start,
                "audio_end": audio_end,
            },
        })
    )
    return path


def _write_cache_manifest(
    cache_dir: Path,
    tokenizer_path: Path,
    *,
    vocab_size: int | None = None,
) -> None:
    from audio_tokenization.token_cache import write_audio_token_cache_manifest

    write_audio_token_cache_manifest(
        cache_dir,
        tokenizer_path=tokenizer_path,
        vocab_size=len(_DummyTokenizer()) if vocab_size is None else vocab_size,
    )


def _write_conversations(path: Path, rows: list[dict], *, row_group_size: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    messages_type = pa.list_(
        pa.struct([
            pa.field("role", pa.string()),
            pa.field("content", pa.string()),
            pa.field(
                "audio",
                pa.list_(
                    pa.struct([
                        pa.field("audio_id", pa.string()),
                        pa.field("source_path", pa.string()),
                    ])
                ),
            ),
        ])
    )
    table = pa.table(
        {
            "sample_id": [row["sample_id"] for row in rows],
            "messages": pa.array([row["messages"] for row in rows], type=messages_type),
            "audio_ids": pa.array([row["audio_ids"] for row in rows], type=pa.list_(pa.string())),
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def _write_media_index(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "audio_id": [row["audio_id"] for row in rows],
        "duration_sec": [row["duration_sec"] for row in rows],
    })
    pq.write_table(table, path)


def test_audio_token_cache_writer_round_trips_multi_asset_spans(tmp_path):
    from audio_tokenization.token_cache import AudioTokenCacheWriter, load_audio_token_cache

    writer = AudioTokenCacheWriter(tmp_path, rank=0)
    writer.add(audio_id="aud-a", tokens=[101, 102], duration_sec=1.25)
    writer.add(audio_id="aud-b", tokens=[201, 202, 203], duration_sec=2.5)
    writer.finalize()

    cache = load_audio_token_cache(tmp_path)

    assert cache.audio_ids == {"aud-a", "aud-b"}
    assert cache.read("aud-a").tolist() == [101, 102]
    assert cache.read("aud-b").tolist() == [201, 202, 203]


def test_audio_token_cache_writer_resumes_after_committed_chunks_and_prunes_orphans(tmp_path):
    from audio_tokenization.token_cache import AudioTokenCacheWriter, load_audio_token_cache

    writer = AudioTokenCacheWriter(tmp_path, rank=0)
    writer.add(audio_id="aud-a", tokens=[101, 102], duration_sec=1.25)
    assert writer.finalize() == 0

    rank_dir = tmp_path / "rank_0000"
    orphan = rank_dir / "audio_tokens.000007.bin"
    orphan.write_bytes(b"orphan")

    writer = AudioTokenCacheWriter(tmp_path, rank=0)
    writer.add(audio_id="aud-b", tokens=[201, 202, 203], duration_sec=2.5)
    assert writer.finalize() == 1

    assert not orphan.exists()
    assert (rank_dir / "audio_index.000000.parquet").is_file()
    assert (rank_dir / "audio_index.000001.parquet").is_file()
    cache = load_audio_token_cache(tmp_path)
    assert cache.read("aud-a").tolist() == [101, 102]
    assert cache.read("aud-b").tolist() == [201, 202, 203]


def test_sft_package_preflight_accepts_structured_audio_attachments(tmp_path):
    from audio_tokenization.sft.preflight import validate_sft_package

    conversations_dir = tmp_path / "conversations"
    _write_conversations(
        conversations_dir / "train.parquet",
        [
            {
                "sample_id": "sample-1",
                "messages": [
                    {
                        "role": "user",
                        "content": "Transcribe this.",
                        "audio": [{"audio_id": "aud-a", "source_path": "a.wav"}],
                    },
                    {"role": "assistant", "content": "hello", "audio": []},
                ],
                "audio_ids": ["aud-a"],
            }
        ],
        row_group_size=1,
    )
    media_dir = tmp_path / "media"
    _write_media_index(media_dir / "_index.parquet", [{"audio_id": "aud-a", "duration_sec": 5.0}])

    report = validate_sft_package(
        conversations_dir=conversations_dir,
        conversations_glob="*.parquet",
        messages_column="messages",
        audio_ids_column="audio_ids",
        audio_placeholder="<audio>",
        media_dir=media_dir,
        media_id_column="audio_id",
        media_duration_column="duration_sec",
        min_duration=1.0,
        max_duration=10.0,
    )

    assert report.conversation_rows == 1
    assert report.conversation_row_groups == 1
    assert report.unique_audio_ids == 1
    assert report.media_rows == 1


def test_sft_package_preflight_rejects_missing_media_ref(tmp_path):
    from audio_tokenization.sft.preflight import validate_sft_package

    conversations_dir = tmp_path / "conversations"
    _write_conversations(
        conversations_dir / "train.parquet",
        [
            {
                "sample_id": "sample-1",
                "messages": [
                    {"role": "user", "content": "<audio>\nWhat is this?", "audio": []},
                    {"role": "assistant", "content": "hello", "audio": []},
                ],
                "audio_ids": ["aud-missing"],
            }
        ],
    )
    media_dir = tmp_path / "media"
    _write_media_index(media_dir / "_index.parquet", [{"audio_id": "aud-a", "duration_sec": 5.0}])

    with pytest.raises(ValueError, match="missing from media parquet"):
        validate_sft_package(
            conversations_dir=conversations_dir,
            conversations_glob="*.parquet",
            messages_column="messages",
            audio_ids_column="audio_ids",
            audio_placeholder="<audio>",
            media_dir=media_dir,
            media_id_column="audio_id",
            media_duration_column="duration_sec",
            min_duration=1.0,
            max_duration=10.0,
        )


def test_sft_package_preflight_rejects_duration_filtered_ref(tmp_path):
    from audio_tokenization.sft.preflight import validate_sft_package

    conversations_dir = tmp_path / "conversations"
    _write_conversations(
        conversations_dir / "train.parquet",
        [
            {
                "sample_id": "sample-1",
                "messages": [
                    {"role": "user", "content": "<audio>\nWhat is this?", "audio": []},
                    {"role": "assistant", "content": "hello", "audio": []},
                ],
                "audio_ids": ["aud-a"],
            }
        ],
    )
    media_dir = tmp_path / "media"
    _write_media_index(media_dir / "_index.parquet", [{"audio_id": "aud-a", "duration_sec": 250.0}])

    with pytest.raises(ValueError, match="duration filters would drop"):
        validate_sft_package(
            conversations_dir=conversations_dir,
            conversations_glob="*.parquet",
            messages_column="messages",
            audio_ids_column="audio_ids",
            audio_placeholder="<audio>",
            media_dir=media_dir,
            media_id_column="audio_id",
            media_duration_column="duration_sec",
            min_duration=1.0,
            max_duration=200.0,
        )


def test_audio_token_cache_loader_rejects_duplicate_audio_ids(tmp_path):
    from audio_tokenization.token_cache import AudioTokenCacheWriter, load_audio_token_cache

    writer = AudioTokenCacheWriter(tmp_path, rank=0)
    writer.add(audio_id="aud-a", tokens=[101], duration_sec=1.0)
    writer.finalize()
    writer = AudioTokenCacheWriter(tmp_path, rank=1)
    writer.add(audio_id="aud-a", tokens=[201], duration_sec=2.0)
    writer.finalize()

    with pytest.raises(ValueError, match="Duplicate audio_id"):
        load_audio_token_cache(tmp_path)


def test_audio_token_cache_loader_rejects_missing_token_file(tmp_path):
    from audio_tokenization.token_cache import AudioTokenCacheWriter, load_audio_token_cache

    writer = AudioTokenCacheWriter(tmp_path, rank=0)
    writer.add(audio_id="aud-a", tokens=[101], duration_sec=1.0)
    writer.finalize()
    next(tmp_path.glob("rank_*/audio_tokens.*.bin")).unlink()

    with pytest.raises(FileNotFoundError, match="missing token file"):
        load_audio_token_cache(tmp_path)


def test_audio_token_cache_loader_rejects_truncated_token_file(tmp_path):
    from audio_tokenization.token_cache import AudioTokenCacheWriter, load_audio_token_cache

    writer = AudioTokenCacheWriter(tmp_path, rank=0)
    writer.add(audio_id="aud-a", tokens=[101, 102], duration_sec=1.0)
    writer.finalize()
    token_path = next(tmp_path.glob("rank_*/audio_tokens.*.bin"))
    with token_path.open("r+b") as f:
        f.truncate(token_path.stat().st_size - 4)

    with pytest.raises(ValueError, match="shorter than its audio-token index requires"):
        load_audio_token_cache(tmp_path)


def test_audio_token_cache_manifest_rejects_tokenizer_mismatch(tmp_path):
    from audio_tokenization.token_cache import (
        validate_audio_token_cache_manifest,
        write_audio_token_cache_manifest,
    )

    tokenizer_a = _write_tokenizer_mapping(tmp_path / "tokenizer-a", audio_start=11)
    tokenizer_b = _write_tokenizer_mapping(tmp_path / "tokenizer-b", audio_start=99)
    write_audio_token_cache_manifest(tmp_path / "cache", tokenizer_path=tokenizer_a, vocab_size=1000)

    with pytest.raises(ValueError, match="tokenizer fingerprint mismatch"):
        validate_audio_token_cache_manifest(tmp_path / "cache", tokenizer_path=tokenizer_b)


def test_audio_cache_handler_writes_manifest(tmp_path):
    from types import SimpleNamespace

    from audio_tokenization.pipelines.lhotse.audio_cache import AudioCacheHandler
    from audio_tokenization.token_cache import (
        read_audio_token_cache_manifest,
        validate_audio_token_cache_manifest,
    )

    tokenizer_path = _write_tokenizer_mapping(tmp_path / "tokenizer")
    spec = SimpleNamespace(tokenizer=SimpleNamespace(path=str(tokenizer_path)))
    handler = AudioCacheHandler(spec)

    class _Tokenizer:
        omni_tokenizer = [0] * 1000

    cache_dir = tmp_path / "cache"
    handler.setup_writer(cache_dir, rank=0, writer_state=0, tokenizer=_Tokenizer())

    manifest = read_audio_token_cache_manifest(cache_dir)
    assert manifest["vocab_size"] == 1000
    validate_audio_token_cache_manifest(cache_dir, tokenizer_path=tokenizer_path)


def test_materialize_sft_rejects_cache_tokenizer_mismatch(tmp_path, monkeypatch):
    from audio_tokenization.token_cache import AudioTokenCacheWriter
    from audio_tokenization.sft.materialize import SftMaterializeConfig, materialize_sft

    tokenizer_a = _write_tokenizer_mapping(tmp_path / "tokenizer-a", audio_start=11)
    tokenizer_b = _write_tokenizer_mapping(tmp_path / "tokenizer-b", audio_start=99)
    cache_dir = tmp_path / "audio_cache"
    writer = AudioTokenCacheWriter(cache_dir, rank=0)
    writer.add(audio_id="aud-a", tokens=[101, 102], duration_sec=1.25)
    writer.finalize()
    _write_cache_manifest(cache_dir, tokenizer_a)

    conversations_dir = tmp_path / "conversations"
    _write_conversations(
        conversations_dir / "train.parquet",
        [
            {
                "sample_id": "sample-1",
                "messages": [
                    {"role": "user", "content": "<audio>\nWhat is this?", "audio": []},
                    {"role": "assistant", "content": "answer", "audio": []},
                ],
                "audio_ids": ["aud-a"],
            }
        ],
    )
    monkeypatch.setattr(
        "audio_tokenization.sft.materialize.load_sft_chat_tokenizer",
        lambda _path: _DummyTokenizer(),
    )

    with pytest.raises(ValueError, match="tokenizer fingerprint mismatch"):
        materialize_sft(
            SftMaterializeConfig(
                conversations_dir=conversations_dir,
                cache_dir=cache_dir,
                output_dir=tmp_path / "out",
                tokenizer_path=tokenizer_b,
            )
        )


def test_materialize_sft_rejects_conversations_missing_cached_audio(tmp_path):
    from audio_tokenization.token_cache import AudioTokenCacheWriter
    from audio_tokenization.sft.materialize import SftMaterializeConfig, materialize_sft

    tokenizer_path = _write_tokenizer_mapping(tmp_path / "tokenizer")
    cache_dir = tmp_path / "audio_cache"
    writer = AudioTokenCacheWriter(cache_dir, rank=0)
    writer.add(audio_id="aud-a", tokens=[101, 102], duration_sec=1.25)
    writer.finalize()
    _write_cache_manifest(cache_dir, tokenizer_path)

    conversations_dir = tmp_path / "conversations"
    _write_conversations(
        conversations_dir / "train.parquet",
        [
            {
                "sample_id": "sample-1",
                "messages": [
                    {
                        "role": "user",
                        "content": "<audio>\nCompare with <audio>.",
                        "audio": [],
                    },
                    {"role": "assistant", "content": "answer", "audio": []},
                ],
                "audio_ids": ["aud-a", "aud-missing"],
            }
        ],
    )

    with pytest.raises(ValueError, match="missing from audio token cache"):
        materialize_sft(
            SftMaterializeConfig(
                conversations_dir=conversations_dir,
                cache_dir=cache_dir,
                output_dir=tmp_path / "out",
                tokenizer_path=tokenizer_path,
            )
        )


def test_sft_cache_reference_validation_uses_audio_ids_column_fast_path(tmp_path, monkeypatch):
    from audio_tokenization.token_cache import AudioTokenCacheWriter, load_audio_token_cache
    import audio_tokenization.sft.materialize as sft_materialize

    cache_dir = tmp_path / "audio_cache"
    writer = AudioTokenCacheWriter(cache_dir, rank=0)
    writer.add(audio_id="aud-a", tokens=[101, 102], duration_sec=1.25)
    writer.finalize()
    cache = load_audio_token_cache(cache_dir)

    conversations_dir = tmp_path / "conversations"
    conversations_dir.mkdir()
    pq.write_table(
        pa.table({
            "sample_id": ["sample-1"],
            "messages": ["this column should not be selected by cache-reference validation"],
            "audio_ids": pa.array([["aud-a", "aud-missing"]], type=pa.list_(pa.string())),
        }),
        conversations_dir / "train.parquet",
    )
    row_groups = [
        sft_materialize._SftRowGroup(
            path=conversations_dir / "train.parquet",
            row_group=0,
            num_rows=1,
        )
    ]
    config = sft_materialize.SftMaterializeConfig(
        conversations_dir=conversations_dir,
        cache_dir=cache_dir,
        output_dir=tmp_path / "out",
        tokenizer_path=tmp_path / "tokenizer",
    )

    original = sft_materialize.select_conversation_columns

    def _record_selected_columns(pf, *, path, columns, required=("sample_id",)):
        assert config.messages_column not in columns
        return original(pf, path=path, columns=columns, required=required)

    monkeypatch.setattr(
        sft_materialize,
        "select_conversation_columns",
        _record_selected_columns,
    )

    with pytest.raises(ValueError, match="missing from audio token cache"):
        sft_materialize._validate_conversation_audio_ids_in_cache(
            row_groups,
            config=config,
            cache=cache,
        )


def test_materialize_sft_assembles_multiple_audio_slots(tmp_path, monkeypatch):
    from audio_tokenization.token_cache import AudioTokenCacheWriter
    from audio_tokenization.sft.materialize import SftMaterializeConfig, materialize_sft

    cache_dir = tmp_path / "audio_cache"
    writer = AudioTokenCacheWriter(cache_dir, rank=0)
    writer.add(audio_id="aud-a", tokens=[101, 102], duration_sec=1.25)
    writer.add(audio_id="aud-b", tokens=[201, 202, 203], duration_sec=2.5)
    writer.finalize()
    tokenizer_path = _write_tokenizer_mapping(tmp_path / "tokenizer")
    _write_cache_manifest(cache_dir, tokenizer_path)

    conversations_dir = tmp_path / "conversations"
    _write_conversations(
        conversations_dir / "train.parquet",
        [
            {
                "sample_id": "sample-1",
                "messages": [
                    {
                        "role": "user",
                        "content": "<audio>\nCompare this with <audio>.",
                        "audio": [
                            {"audio_id": "aud-a", "source_path": "a.wav"},
                            {"audio_id": "aud-b", "source_path": "b.wav"},
                        ],
                    },
                    {"role": "assistant", "content": "The second clip is longer.", "audio": []},
                ],
                "audio_ids": ["aud-a", "aud-b"],
            }
        ],
    )
    monkeypatch.setattr(
        "audio_tokenization.sft.materialize.load_sft_chat_tokenizer",
        lambda _path: _DummyTokenizer(),
    )

    output_dir = tmp_path / "out"
    result = materialize_sft(
        SftMaterializeConfig(
            conversations_dir=conversations_dir,
            cache_dir=cache_dir,
            output_dir=output_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=4096,
        )
    )

    assert result["samples_processed"] == 1
    pairs = collect_cutid_token_pairs(output_dir)
    tokens = list(pairs["sample-1"])
    assert tokens[0] == _DummyTokenizer.bos_token_id
    assert tokens[-1] == _DummyTokenizer.eos_token_id
    pos_a = _find_subsequence(tokens, [101, 102])
    pos_b = _find_subsequence(tokens, [201, 202, 203])
    assert 0 < pos_a < pos_b


def test_materialize_sft_renders_structured_audio_attachments(tmp_path, monkeypatch):
    from audio_tokenization.token_cache import AudioTokenCacheWriter
    from audio_tokenization.sft.materialize import SftMaterializeConfig, materialize_sft

    cache_dir = tmp_path / "audio_cache"
    writer = AudioTokenCacheWriter(cache_dir, rank=0)
    writer.add(audio_id="aud-a", tokens=[101, 102], duration_sec=1.25)
    writer.finalize()
    tokenizer_path = _write_tokenizer_mapping(tmp_path / "tokenizer")
    _write_cache_manifest(cache_dir, tokenizer_path)

    conversations_dir = tmp_path / "conversations"
    _write_conversations(
        conversations_dir / "train.parquet",
        [
            {
                "sample_id": "structured-audio",
                "messages": [
                    {
                        "role": "user",
                        "content": "Please transcribe this recording.",
                        "audio": [{"audio_id": "aud-a", "source_path": "a.wav"}],
                    },
                    {"role": "assistant", "content": "hello world", "audio": []},
                ],
                "audio_ids": ["aud-a"],
            }
        ],
    )
    monkeypatch.setattr(
        "audio_tokenization.sft.materialize.load_sft_chat_tokenizer",
        lambda _path: _DummyTokenizer(),
    )

    result = materialize_sft(
        SftMaterializeConfig(
            conversations_dir=conversations_dir,
            cache_dir=cache_dir,
            output_dir=tmp_path / "out",
            tokenizer_path=tokenizer_path,
            max_seq_len=4096,
        )
    )

    assert result["samples_processed"] == 1
    tokens = list(collect_cutid_token_pairs(tmp_path / "out")["structured-audio"])
    assert _find_subsequence(tokens, [101, 102]) > 0


def test_assemble_sft_conversation_avoids_torch_tensor_copies(monkeypatch):
    from audio_tokenization.sft import materialize as sft_materialize
    from audio_tokenization.sft.materialize import assemble_sft_conversation

    class _Cache:
        def read(self, audio_id: str):
            assert audio_id == "aud-a"
            return np.array([101, 102], dtype=np.int32)

    if hasattr(sft_materialize, "torch"):
        monkeypatch.setattr(
            sft_materialize.torch,
            "tensor",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("torch.tensor copy")),
        )

    tokens = assemble_sft_conversation(
        sample_id="sample-1",
        messages=[
            {"role": "user", "content": "<audio>\nWhat is this?", "audio": []},
            {"role": "assistant", "content": "A spoken question.", "audio": []},
        ],
        audio_ids=["aud-a"],
        cache=_Cache(),
        tokenizer=_DummyTokenizer(),
    )

    assert isinstance(tokens, np.ndarray)
    assert tokens[0] == _DummyTokenizer.bos_token_id
    assert tokens[-1] == _DummyTokenizer.eos_token_id
    assert _find_subsequence(tokens.tolist(), [101, 102]) > 0


def test_materialize_sft_parallelizes_by_sft_row_group(tmp_path, monkeypatch):
    from audio_tokenization.token_cache import AudioTokenCacheWriter
    from audio_tokenization.sft import materialize as sft_materialize
    from audio_tokenization.sft.materialize import SftMaterializeConfig, materialize_sft

    cache_dir = tmp_path / "audio_cache"
    writer = AudioTokenCacheWriter(cache_dir, rank=0)
    for idx in range(4):
        writer.add(audio_id=f"aud-{idx}", tokens=[100 + idx], duration_sec=1.0)
    writer.finalize()
    tokenizer_path = _write_tokenizer_mapping(tmp_path / "tokenizer")
    _write_cache_manifest(cache_dir, tokenizer_path)

    rows = []
    for idx in range(4):
        rows.append(
            {
                "sample_id": f"sample-{idx}",
                "messages": [
                    {"role": "user", "content": "<audio>\nWhat is this?", "audio": []},
                    {"role": "assistant", "content": f"clip {idx}", "audio": []},
                ],
                "audio_ids": [f"aud-{idx}"],
            }
        )
    conversations_dir = tmp_path / "conversations"
    _write_conversations(conversations_dir / "train.parquet", rows, row_group_size=1)
    monkeypatch.setattr(
        "audio_tokenization.sft.materialize.load_sft_chat_tokenizer",
        lambda _path: _DummyTokenizer(),
    )
    load_calls_path = tmp_path / "cache_load_calls.txt"
    original_load_audio_token_cache = sft_materialize.load_audio_token_cache

    def _counting_load_audio_token_cache(path):
        with load_calls_path.open("a") as f:
            f.write("load\n")
        return original_load_audio_token_cache(path)

    monkeypatch.setattr(
        "audio_tokenization.sft.materialize.load_audio_token_cache",
        _counting_load_audio_token_cache,
    )

    output_dir = tmp_path / "out"
    result = materialize_sft(
        SftMaterializeConfig(
            conversations_dir=conversations_dir,
            cache_dir=cache_dir,
            output_dir=output_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=4096,
            num_workers=2,
        )
    )

    assert result["samples_processed"] == 4
    assert result["num_workers"] == 2
    assert result["chunks_written"] == 2
    assert sorted(path.name for path in output_dir.glob("*.idx")) == [
        "rank_0000_chunk_0000.idx",
        "rank_0001_chunk_0000.idx",
    ]
    assert load_calls_path.read_text().splitlines() == ["load"]
    pairs = collect_cutid_token_pairs(output_dir)
    assert set(pairs) == {f"sample-{idx}" for idx in range(4)}
    for idx in range(4):
        assert 100 + idx in pairs[f"sample-{idx}"]


def test_materialize_sft_routes_by_seq_threshold(tmp_path, monkeypatch):
    from audio_tokenization.token_cache import AudioTokenCacheWriter
    from audio_tokenization.sft.materialize import SftMaterializeConfig, materialize_sft

    cache_dir = tmp_path / "audio_cache"
    writer = AudioTokenCacheWriter(cache_dir, rank=0)
    writer.add(audio_id="aud-short", tokens=[101], duration_sec=1.0)
    writer.add(audio_id="aud-long", tokens=[201], duration_sec=1.0)
    writer.finalize()
    tokenizer_path = _write_tokenizer_mapping(tmp_path / "tokenizer")
    _write_cache_manifest(cache_dir, tokenizer_path)

    conversations_dir = tmp_path / "conversations"
    _write_conversations(
        conversations_dir / "train.parquet",
        [
            {
                "sample_id": "short-sample",
                "messages": [
                    {"role": "user", "content": "<audio>\nx", "audio": []},
                    {"role": "assistant", "content": "y", "audio": []},
                ],
                "audio_ids": ["aud-short"],
            },
            {
                "sample_id": "long-sample",
                "messages": [
                    {"role": "user", "content": "<audio>\n" + ("x" * 200), "audio": []},
                    {"role": "assistant", "content": "y", "audio": []},
                ],
                "audio_ids": ["aud-long"],
            },
        ],
    )
    monkeypatch.setattr(
        "audio_tokenization.sft.materialize.load_sft_chat_tokenizer",
        lambda _path: _DummyTokenizer(),
    )

    output_dir = tmp_path / "out"
    result = materialize_sft(
        SftMaterializeConfig(
            conversations_dir=conversations_dir,
            cache_dir=cache_dir,
            output_dir=output_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=4096,
            seq_threshold=80,
        )
    )

    assert result["stage2_samples"] == 1
    assert result["lct_samples"] == 1
    assert set(collect_cutid_token_pairs(output_dir / "stage2")) == {"short-sample"}
    assert set(collect_cutid_token_pairs(output_dir / "lct")) == {"long-sample"}


def test_materialize_sft_fails_loudly_on_placeholder_audio_mismatch(tmp_path, monkeypatch):
    from audio_tokenization.token_cache import AudioTokenCacheWriter
    from audio_tokenization.sft.materialize import SftMaterializeConfig, materialize_sft

    cache_dir = tmp_path / "audio_cache"
    writer = AudioTokenCacheWriter(cache_dir, rank=0)
    writer.add(audio_id="aud-a", tokens=[101, 102], duration_sec=1.25)
    writer.finalize()
    tokenizer_path = _write_tokenizer_mapping(tmp_path / "tokenizer")
    _write_cache_manifest(cache_dir, tokenizer_path)

    conversations_dir = tmp_path / "conversations"
    _write_conversations(
        conversations_dir / "train.parquet",
        [
            {
                "sample_id": "bad-sample",
                "messages": [
                    {"role": "user", "content": "<audio> and <audio>", "audio": []},
                    {"role": "assistant", "content": "answer", "audio": []},
                ],
                "audio_ids": ["aud-a"],
            }
        ],
    )
    monkeypatch.setattr(
        "audio_tokenization.sft.materialize.load_sft_chat_tokenizer",
        lambda _path: _DummyTokenizer(),
    )

    with pytest.raises(ValueError, match="has 2 audio placeholders but 1 audio ids"):
        materialize_sft(
            SftMaterializeConfig(
                conversations_dir=conversations_dir,
                cache_dir=cache_dir,
                output_dir=tmp_path / "out",
                tokenizer_path=tokenizer_path,
            )
        )


def test_materialize_sft_requires_explicit_audio_placeholder(tmp_path, monkeypatch):
    from audio_tokenization.token_cache import AudioTokenCacheWriter
    from audio_tokenization.sft.materialize import SftMaterializeConfig, materialize_sft

    cache_dir = tmp_path / "audio_cache"
    writer = AudioTokenCacheWriter(cache_dir, rank=0)
    writer.add(audio_id="aud-a", tokens=[101, 102], duration_sec=1.25)
    writer.finalize()
    tokenizer_path = _write_tokenizer_mapping(tmp_path / "tokenizer")
    _write_cache_manifest(cache_dir, tokenizer_path)

    conversations_dir = tmp_path / "conversations"
    _write_conversations(
        conversations_dir / "train.parquet",
        [
            {
                "sample_id": "missing-placeholder",
                "messages": [
                    {"role": "user", "content": "What is this clip?", "audio": []},
                    {"role": "assistant", "content": "answer", "audio": []},
                ],
                "audio_ids": ["aud-a"],
            }
        ],
    )
    monkeypatch.setattr(
        "audio_tokenization.sft.materialize.load_sft_chat_tokenizer",
        lambda _path: _DummyTokenizer(),
    )

    with pytest.raises(ValueError, match="has 0 audio placeholders but 1 audio ids"):
        materialize_sft(
            SftMaterializeConfig(
                conversations_dir=conversations_dir,
                cache_dir=cache_dir,
                output_dir=tmp_path / "out",
                tokenizer_path=tokenizer_path,
            )
        )


def test_materialize_sft_accepts_json_messages_column(tmp_path, monkeypatch):
    from audio_tokenization.token_cache import AudioTokenCacheWriter
    from audio_tokenization.sft.materialize import SftMaterializeConfig, materialize_sft

    cache_dir = tmp_path / "audio_cache"
    writer = AudioTokenCacheWriter(cache_dir, rank=0)
    writer.add(audio_id="aud-a", tokens=[101, 102], duration_sec=1.25)
    writer.finalize()
    tokenizer_path = _write_tokenizer_mapping(tmp_path / "tokenizer")
    _write_cache_manifest(cache_dir, tokenizer_path)

    conversations_dir = tmp_path / "conversations"
    messages = [
        {"role": "user", "content": "<audio>\nWhat is this?", "audio": []},
        {"role": "assistant", "content": "A spoken question.", "audio": []},
    ]
    conversations_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "sample_id": ["json-sample"],
                "messages_json": [json.dumps(messages)],
                "audio_ids": pa.array([["aud-a"]], type=pa.list_(pa.string())),
            }
        ),
        conversations_dir / "train.parquet",
    )
    monkeypatch.setattr(
        "audio_tokenization.sft.materialize.load_sft_chat_tokenizer",
        lambda _path: _DummyTokenizer(),
    )

    result = materialize_sft(
        SftMaterializeConfig(
            conversations_dir=conversations_dir,
            cache_dir=cache_dir,
            output_dir=tmp_path / "out",
            tokenizer_path=tokenizer_path,
            messages_column="messages_json",
        )
    )

    assert result["samples_processed"] == 1
    tokens = list(collect_cutid_token_pairs(tmp_path / "out")["json-sample"])
    assert _find_subsequence(tokens, [101, 102]) > 0


def test_materialize_sft_sizes_output_dtype_from_audio_cache_vocab(tmp_path, monkeypatch):
    from audio_tokenization.token_cache import AudioTokenCacheWriter
    from audio_tokenization.sft.materialize import SftMaterializeConfig, materialize_sft

    audio_token_id = 131073
    cache_dir = tmp_path / "audio_cache"
    writer = AudioTokenCacheWriter(cache_dir, rank=0)
    writer.add(audio_id="aud-a", tokens=[audio_token_id], duration_sec=1.25)
    writer.finalize()
    tokenizer_path = _write_tokenizer_mapping(tmp_path / "tokenizer")
    _write_cache_manifest(cache_dir, tokenizer_path, vocab_size=266000)

    conversations_dir = tmp_path / "conversations"
    _write_conversations(
        conversations_dir / "train.parquet",
        [
            {
                "sample_id": "high-audio-token",
                "messages": [
                    {"role": "user", "content": "<audio>\nWhat is this?", "audio": []},
                    {"role": "assistant", "content": "answer", "audio": []},
                ],
                "audio_ids": ["aud-a"],
            }
        ],
    )
    monkeypatch.setattr(
        "audio_tokenization.sft.materialize.load_sft_chat_tokenizer",
        lambda _path: _DummyTokenizer(),
    )

    materialize_sft(
        SftMaterializeConfig(
            conversations_dir=conversations_dir,
            cache_dir=cache_dir,
            output_dir=tmp_path / "out",
            tokenizer_path=tokenizer_path,
        )
    )

    tokens = list(collect_cutid_token_pairs(tmp_path / "out")["high-audio-token"])
    assert audio_token_id in tokens


def _find_subsequence(tokens: list[int], needle: list[int]) -> int:
    for start in range(0, len(tokens) - len(needle) + 1):
        if tokens[start : start + len(needle)] == needle:
            return start
    raise AssertionError(f"{needle!r} not found in token sequence")
