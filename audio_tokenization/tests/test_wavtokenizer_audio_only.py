from types import SimpleNamespace

import torch

from audio_tokenization.config.schema import TokenizeSpec
from audio_tokenization.pipelines.lhotse.audio_only import AudioOnlyHandler
from audio_tokenization.pipelines.lhotse.checkpoint import WorkerStats
from audio_tokenization.vokenizers.wavtokenizer.audio_only import WavTokenizerAudioOnly


class _FakeWavTokenizer:
    downsample_rate = 10

    def __init__(self):
        self.encoded_lengths = []

    def encode(self, audio, sr):
        self.encoded_lengths.append(audio.shape[1])
        token_count = (audio.shape[1] + self.downsample_rate - 1) // self.downsample_rate
        rows = []
        for row in range(audio.shape[0]):
            rows.append(torch.arange(token_count, device=audio.device) + row * 1000)
        return torch.stack(rows, dim=0), {}


def _tokenizer(trim_last_tokens=2):
    tok = WavTokenizerAudioOnly.__new__(WavTokenizerAudioOnly)
    tok.device = torch.device("cpu")
    tok.trim_last_tokens = trim_last_tokens
    tok._wavtokenizer = _FakeWavTokenizer()
    tok._bos_id = 1
    tok._eos_id = 2
    tok._audio_start_id = 3
    tok._audio_end_id = 4
    tok._audio_token_offset = 100
    return tok


class _FakeBuilder:
    def __init__(self):
        self.items = []
        self.documents = 0

    def add_item(self, tensor):
        self.items.append(tensor.clone())

    def end_document(self):
        self.documents += 1


class _FakeCutIds:
    def __init__(self):
        self.ids = []

    def write(self, cut_id):
        self.ids.append(cut_id)


def test_trim_last_tokens_is_independent_of_batch_padding_status():
    tok = _tokenizer(trim_last_tokens=2)

    outputs = tok.tokenize_batch(
        torch.zeros(4, 130),
        sample_rate=24_000,
        orig_audio_samples=[80, 100, 130, 121],
        pad_audio_samples=130,
    )

    # Every cut loses the same configured boundary trim. This keeps
    # cut_id -> tokens invariant across different distributed batch layouts.
    assert len(outputs[0]) == 1 + 1 + 6 + 1 + 1   # ceil(80 / 10) - 2
    assert len(outputs[1]) == 1 + 1 + 8 + 1 + 1   # ceil(100 / 10) - 2
    assert len(outputs[2]) == 1 + 1 + 11 + 1 + 1  # ceil(130 / 10) - 2
    assert len(outputs[3]) == 1 + 1 + 11 + 1 + 1  # ceil(121 / 10) - 2
    assert tok.wavtokenizer.encoded_lengths == [130]


def test_audio_only_handler_applies_deterministic_boundary_trim():
    """The production handler must not make trim depend on local batch shape."""
    tok = _tokenizer(trim_last_tokens=2)
    handler = AudioOnlyHandler(
        TokenizeSpec.model_validate({
            "tokenizer": {"path": "/tmp/tokenizer"},
            "output": {"output_dir": "/tmp/out"},
        })
    )
    handler._builder = _FakeBuilder()
    handler._cut_ids = _FakeCutIds()
    stats = WorkerStats()

    handler.process_batch(
        {
            "audio": torch.zeros(4, 130),
            "audio_lens": torch.tensor([80, 100, 130, 121], dtype=torch.int64),
            "cuts": [SimpleNamespace(id=f"cut-{i}") for i in range(4)],
        },
        tok,
        stats,
        target_sr=24_000,
        device="cpu",
    )

    # Remove BOS/audio_start/audio_end/EOS to inspect produced codec-token counts.
    audio_token_counts = [len(item) - 4 for item in handler._builder.items]
    assert audio_token_counts == [6, 8, 11, 11]
    assert handler._builder.documents == 4
    assert handler._cut_ids.ids == ["cut-0", "cut-1", "cut-2", "cut-3"]
    assert stats.samples_processed == 4
