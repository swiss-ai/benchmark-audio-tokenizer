"""Shared LLM-based inverse text normalization (ITN) utilities.

Provides the system prompt, batch inference, and validation for
converting spoken-form numbers to digits, capitalizing proper nouns, etc.
"""

from __future__ import annotations

ITN_SYSTEM_PROMPT = """\
Normalize the text by applying these rules:
1. Convert spoken numbers to digits precisely: "six thousand three hundred and thirty three" → "6333". Do NOT round.
2. Capitalize proper nouns: "barack obama" → "Barack Obama", "new york" → "New York"
3. Capitalize acronyms: "u s" → "US", "n b a" → "NBA"
4. Normalize dates: "january fifth two thousand twenty" → "January 5th, 2020"
5. Normalize ordinals: "twenty first" → "21st"
6. Normalize measurements: "five point two kilograms" → "5.2 kilograms"
7. Normalize times: "three thirty p m" → "3:30 PM"
Do NOT add, remove, duplicate, or rephrase any words. \
Do NOT capitalize the first word unless it is a proper noun. \
Keep currency as words: "5 dollars" NOT "$5". Never use currency symbols. \
If the text already contains $ signs, keep them exactly as-is. \
If nothing needs changing, return the input exactly as-is.

Input: sold it for twenty four million dollars, but barack obama said two examples.
Output: sold it for 24 million dollars, but Barack Obama said 2 examples.
Input: as of two thousand and eighteen, three thousand nine hundred and fifty five people lived in new york.
Output: as of 2018, 3955 people lived in New York.
Input: six thousand three hundred and thirty three students need to ride.
Output: 6333 students need to ride.
Input: one thousand nine hundred and twenty miles by the way of forts kearney.
Output: 1920 miles by the way of Forts Kearney.
Input: i own myself a five hundred dollar man and two thousand dollars' worth of family.
Output: i own myself a 500 dollar man and 2000 dollars' worth of family.
Input: from around the eighteen ninety s to the nineteen fifty s.
Output: from around the 1890s to the 1950s.
Input: the n b a finals were on january twenty first, two thousand twenty.
Output: the NBA finals were on January 21st, 2020.
Input: seventy thousand dollars is not that much labor.
Output: 70,000 dollars is not that much labor.
Input: the court ruled in favor of the defendant last tuesday.
Output: the court ruled in favor of the defendant last Tuesday."""


# Phrases from the prompt that should never appear in real output
_PROMPT_LEAKS = frozenset({
    "the court ruled in favor of the defendant last tuesday",
    "the input text does not",
    "does not require normalization",
    "according to the specified rules",
})


def validate_itn(original: str, normalized: str) -> str:
    """Return original if the LLM output looks bad."""
    if not normalized:
        return original
    if len(normalized) > len(original) * 1.5:
        return original
    if len(normalized) < len(original) * 0.3:
        return original
    if "Input:" in normalized or "Output:" in normalized:
        return original
    # Check for prompt example leaking into output
    norm_lower = normalized.lower()
    for phrase in _PROMPT_LEAKS:
        if phrase in norm_lower and phrase not in original.lower():
            return original
    return normalized


def itn_batch(llm, tokenizer, sampling_params, texts: list[str]) -> list[str]:
    """Run ITN on a batch of texts via vLLM offline inference."""
    conversations = [
        [
            {"role": "system", "content": ITN_SYSTEM_PROMPT},
            {"role": "user", "content": f"Input: {t}\nOutput:"},
        ]
        for t in texts
    ]
    prompts = [
        tokenizer.apply_chat_template(
            c, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for c in conversations
    ]
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for orig, out in zip(texts, outputs):
        gen = out.outputs[0].text.strip()
        if "</think>" in gen:
            gen = gen.split("</think>")[-1].strip()
        results.append(validate_itn(orig, gen))
    return results


def load_llm(model_path: str, temperature: float = 0.0, max_tokens: int = 256):
    """Load vLLM model + tokenizer + sampling params for ITN."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path, trust_remote_code=True,
        tensor_parallel_size=1, max_model_len=1024, dtype="auto",
    )
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens, stop=["\n"],
    )
    return llm, tokenizer, sampling_params


def write_jsonl_zst(entries: list[dict], out_path):
    """Write entries to a zstandard-compressed JSONL file."""
    import json
    from pathlib import Path

    from audio_tokenization.utils.io import atomic_streaming_write

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with atomic_streaming_write(out_path, compression="zst") as f:
        for e in entries:
            f.write((json.dumps(e, ensure_ascii=False) + "\n").encode("utf-8"))
