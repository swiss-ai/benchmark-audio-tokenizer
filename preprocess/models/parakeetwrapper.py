import logging
from typing import List

import torch
import torchaudio

logger = logging.getLogger(__name__)

NEMO_SR = 16000

class ParakeetWrapper:
    """Batch transcription for European languages"""

    def __init__(self,
                 rank: int,
                 device: str = "cuda", 
                 model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
                 cache_dir: str | None = None
                 ):
        import nemo.collections.asr as nemo_asr
        import os

        if cache_dir is not None:
            os.environ["NEMO_HOME"] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)

        self.device = device
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        self.model.to(device).eval()
        self._resampler_cache: Dict[int, torchaudio.transforms.Resample] = {}
        logger.info(f"[rank {rank}] Parakeet loaded: {model_id} on {device}")
    
    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        """Cache torchaudio resamplers keyed by source sample rate."""
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(
                orig_sr, NEMO_SR
            )
        return self._resampler_cache[orig_sr]

    @torch.inference_mode()
    def transcribe_batch(
            self,
            audios: torch.Tensor,
            audio_lens: torch.Tensor,
            sr: int,
            **kwargs
        ) -> List[str]:
        """Transcribe a batch of waveforms.
 
        Args:
            audios: (B, T) padded waveforms.
            audio_lens: (B,) original sample lengths.

        Returns:
            List of transcription strings, one per sample.
        """
        audios = audios.to(self.device)
        audio_lens = audio_lens.to(self.device)

        if sr != NEMO_SR:
            resampler = self._get_resampler(sr)
            audios_16k = resampler(audios)
            scale = NEMO_SR / sr
            audio_lens = (audio_lens.float() * scale).long()
        else:
            audios_16k = audios

        features, features_len = self.model.preprocessor(
                                                        input_signal=audios_16k, 
                                                        length=audio_lens  
                                                        )
        
        encoded, encoded_len = self.model.encoder(
                                    audio_signal=features, 
                                    length=features_len 
                                )

        output = self.model.decoding.decode_predictions_tensor(encoded, encoded_len)
        transcriptions = [op[0].text if isinstance(op, list) else op.text for op in output]
        return transcriptions