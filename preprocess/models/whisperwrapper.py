import logging
from typing import Dict, List, Optional

import torch
import torchaudio

logger = logging.getLogger(__name__)

WHISPER_SR = 16000

class WhisperWrapper:
    """Batch language detection and transcription"""

    def __init__(self, 
                 rank: int,
                 device: str = "cuda", 
                 model_id: str = "openai/whisper-large-v3",
                 cache_dir: str | None = None
                 ):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from whisper_normalizer.basic import BasicTextNormalizer
        from torchmetrics.text import CharErrorRate

        self.device = device

        self.normalizer = BasicTextNormalizer()
        self.cer_calc = CharErrorRate()

        self.model_id = model_id
        self.processor = WhisperProcessor.from_pretrained(model_id, cache_dir = cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir = cache_dir)
        self.model.to(device).eval()
        self._resampler_cache: Dict[int, torchaudio.transforms.Resample] = {}
        logger.info(f"[rank {rank}] Whisper loaded: {model_id} on {device}")

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        """Cache torchaudio resamplers keyed by source sample rate."""
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(
                orig_sr, WHISPER_SR
            )
        return self._resampler_cache[orig_sr]

    def _prepare_features(
            self,
            audios: torch.Tensor,
            audio_lens: torch.Tensor,
            sr: int,
        ) -> torch.Tensor:
        """Resample and extract Whisper log-mel features.

        Args:
            audios: (B, T) padded waveforms on any device.
            audio_lens: (B,) original sample counts (before padding).
            sr: Sample rate of *audios*.
 
        Returns:
            input_features: (B, 128, 3000) on ``self.device``.
        """
        # Resample
        if sr != WHISPER_SR:
            resampler = self._get_resampler(sr)
            audios_16k = resampler(audios)
            scale = WHISPER_SR / sr
            audio_lens = (audio_lens.float() * scale).long()
        else:
            audios_16k = audios
        
        # Trim padding per sample and pass through the processor.
        audios_16k_cpu = audios_16k.cpu()
        lens_cpu = audio_lens.cpu()
        wavs = [audios_16k_cpu[i, : int(lens_cpu[i])].numpy() for i in range(audios_16k_cpu.shape[0])]
        
        features = self.processor(
            wavs,
            sampling_rate=WHISPER_SR,
            return_tensors="pt",
            padding="max_length",
        )
        return features.input_features.to(self.device)
    
    @torch.inference_mode()
    def detect_language(
        self,
        audios: torch.Tensor,
        audio_lens: torch.Tensor,
        sr: int = 16000,
    ) -> List[Dict[str, object]]:
        """Detect language for a batch of waveforms.

        Uses Whisper's generate() with max_new_tokens=1 to predict
        just the language token, then decodes it.

        Args:
            audios: (B, T) padded waveforms.
            audio_lens: (B,) original sample lengths.
            sr: Source sample rate.

        Returns:
            List of {"language": str, "language_probability": float},
            one per sample.
        """
        input_features = self._prepare_features(audios, audio_lens, sr)
        B = input_features.shape[0]
        results: List[Dict[str, object]] = []

        # Get all language tokens from the tokenizer
        tokenizer = self.processor.tokenizer
        lang_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
        lang_token_ids = tokenizer.convert_tokens_to_ids(lang_tokens)

        for i in range(B):
            feat_i = input_features[i : i + 1]

            # Get decoder input: <|startoftranscript|>
            decoder_input_ids = torch.tensor(
                [[self.model.config.decoder_start_token_id]]
            ).to(self.device)

            # Forward pass to get logits
            outputs = self.model(
                input_features=feat_i,
                decoder_input_ids=decoder_input_ids,
            )

            # Extract logits at the language token position
            logits = outputs.logits[0, -1]  # (vocab_size,)
            lang_logits = logits[lang_token_ids]
            lang_probs = torch.softmax(lang_logits, dim=-1)

            best_idx = lang_probs.argmax().item()
            best_lang_token = lang_tokens[best_idx]
            # Strip <| and |> to get language code
            best_lang = best_lang_token.strip("<|>")
            best_prob = lang_probs[best_idx].item()

            results.append({
                "language": best_lang,
                "language_probability": round(best_prob, 4),
            })

        return results


    @torch.inference_mode()
    def transcribe_batch(
            self,
            audios: torch.Tensor,
            audio_lens: torch.Tensor,
            sr: int = 16000,
            language: Optional[str] = None,
        ) -> List[str]:
        """Transcribe a batch of waveforms.
 
        Args:
            audios: (B, T) padded waveforms.
            audio_lens: (B,) original sample lengths.
            sr: Source sample rate.
            languages: Per-sample language codes (from :meth:`detect_language_batch`).
                If ``None``, Whisper auto-detects per sample.
 
        Returns:
            List of transcription strings, one per sample.
        """
        input_features = self._prepare_features(audios, audio_lens, sr)
 
        forced_decoder_ids = None
        if language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, task="transcribe",
            )
        generated_ids = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
        )
        texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )
        return [t.strip() for t in texts]

    def calculate_cer(self, 
                      *,
                      transcriptions: List[str],
                      references: List[str]
                      ) -> List[float]:
        """
        Calculates the normalized Character Error Rate (CER).
        """

        norm_trans = [self.normalizer(text) for text in transcriptions]
        norm_refs  = [self.normalizer(text) for text in references]

        cer_scores = [
            self.cer_calc(preds=t, target=r).item() 
            for t, r in zip(norm_trans, norm_refs)
        ]
        return cer_scores
