import os
import sys
import torch
import torchaudio
import numpy as np
import logging 
from typing import Optional


varstok_path = os.path.join(os.path.dirname(__file__), '..', '..', 'repos', 'FunResearch','VARSTok')
sys.path.insert(0, varstok_path)

from ..base import BaseAudioTokenizer

logger = logging.getLogger(__name__)

class VARSTokWrapper(BaseAudioTokenizer):

    name = "varstok"
    
    def __init__(self, device: str = "cuda", checkpoint: Optional[str] = None):
        self.device = device
        self.checkpoint = checkpoint

        self.tokens_per_second = 36.81
        self._load_model()


    def _load_model(self):
        cache_dir = "/capstor/store/cscs/swissai/infra01/MLLM/varstok"
        os.makedirs(cache_dir, exist_ok=True)

        model_name = "varstok_thres08_maxspan4.ckpt"
        config_name = "varstok_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"

        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="ZhengRachel/VARSTok",
            filename=model_name,
            cache_dir=cache_dir,
            local_dir=cache_dir
        )
        from decoder.pretrained import VARSTok
        config_path = os.path.join(os.path.dirname(__file__), '..','..', 'configs', config_name)
        self.model = VARSTok.from_pretrained(config_path, model_path).to(self.device)

    @property
    def sample_rate(self) -> int:
        """Input sample rate for the tokenizer."""
        return 24000  

    @sample_rate.setter
    def sample_rate(self, value: int):
        """Setter for sample_rate (required by base class)."""
        if value != 24000:
            logger.warning(f"VARSTok uses 24kHz sample rate, ignoring requested {value}Hz")

    @property
    def output_sample_rate(self) -> int:
        """Output sample rate for the decoder."""
        return 24000  

    @output_sample_rate.setter
    def output_sample_rate(self, value: int):
        """Setter for output_sample_rate (for consistency)."""
        if value != 24000:
            logger.warning(f"VARSTok uses fixed 24kHz output rate, ignoring requested {value}Hz")

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return 4096  # VARSTok uses 4096 codes

    @property
    def downsample_rate(self) -> int:
        """Downsampling rate from audio samples to tokens."""
        return 652  # 24000 / 36.81 = 652 (~36.81 tokens per second)


    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        # Ensure correct shape
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # Remove channel dimension
        
        from encoder.utils import convert_audio
        audio = convert_audio(audio, self.sample_rate, 24000, 1) 
        audio = audio.to(self.device)

        bandwidth_id = torch.tensor([0]).to(self.device)
        tokens, codes, cluster_lengths = self.model.encode_infer(audio, bandwidth_id=bandwidth_id)


        codes.cluster_lengths = cluster_lengths
        codes.tokens = tokens
        return codes

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        bandwidth_id = torch.tensor([0]).to(self.device)
        
        features =  tokens.tokens.to(self.device)
        cluster_lengths = tokens.cluster_lengths.to(self.device)
        audio = self.model.decode(features, cluster_lengths, bandwidth_id=bandwidth_id) 

        # Ensure output shape is (B, 1, T)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        return audio

 