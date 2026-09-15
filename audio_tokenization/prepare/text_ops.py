"""Text tokenizer helpers for prepare scripts."""

from __future__ import annotations

import logging
from pathlib import Path


def load_text_tokenizer(tokenizer_path: str | Path):
    """Load a Rust fast tokenizer from a tokenizer.json file."""
    if tokenizer_path is None:
        return None
    from tokenizers import Tokenizer

    path = Path(tokenizer_path)
    if not path.is_file():
        raise FileNotFoundError(f"Text tokenizer not found: {path}")
    tok = Tokenizer.from_file(str(path))
    logging.getLogger(__name__).info(f"Text pre-tokenization enabled: {path}")
    return tok


def make_text_tokenize_fn(tokenizer, extra_custom_columns=None):
    """Return a lhotse cut map function that tokenizes supervision text."""
    _logger = logging.getLogger(__name__)
    _extra = tuple(extra_custom_columns or ())

    def _tokenize_text(cut):
        texts = [s.text for s in (cut.supervisions or []) if s.text]
        if not texts:
            return cut
        if len(texts) > 1:
            _logger.debug(
                "Cut %s: merging %d supervision texts into one", cut.id, len(texts)
            )
        ids = tokenizer.encode(" ".join(texts), add_special_tokens=False).ids
        cut.custom = cut.custom or {}
        cut.custom["text_tokens"] = ids
        for col in _extra:
            val = cut.custom.get(col)
            if val and isinstance(val, str):
                cut.custom[f"{col}_tokens"] = tokenizer.encode(
                    val, add_special_tokens=False
                ).ids
        return cut

    return _tokenize_text
