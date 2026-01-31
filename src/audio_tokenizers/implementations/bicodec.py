import torch
import logging
import os
import sys
import time
from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np

sparktts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'repos', 'Spark-TTS')
sys.path.insert(0, sparktts_path)

from sparktts.utils.file import load_config
from sparktts.utils.audio import audio_volume_normalize
from sparktts.models.audio_tokenizer import BiCodecTokenizer

from ..base import BaseAudioTokenizer

logger = logging.getLogger(__name__)

class BiCodecAudioTokenizer(BaseAudioTokenizer):
    """
    Wrapper for BiCodec - discrete codec with fixed 50 tokens per second.

    BiCodec uses a Wav2Vec encoder with Vector Quantization for semantic tokens
    and an ECAPA-Time Delay NN for global tokens. Semantic tokens are fixed at
    50 tokens per second that captures linguistic content while global tokens are
    fixed number (32 tokens).
    Semantic codebook size is 8192 while global codebook size is 4096. The global
    token is only used in 
    """

    name = "bicodec"

    def _load_model(self):
        """Load the WavTokenizer model."""
        try:
            # Model directory
            cache_dir = "/capstor/store/cscs/swissai/infra01/MLLM/SparkTTS-0.5B"
            os.makedirs(cache_dir, exist_ok = True) 

            # Download from HuggingFace if not cached
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="SparkAudio/Spark-TTS-0.5B", 
                cache_dir=cache_dir,
                local_dir=cache_dir
                )
            
            config_path = os.path.join(cache_dir,"config.yaml")
            
            logger.info(f"Loading BiCodec from {cache_dir}")
            logger.info(f"Using config: {config_path}")

            config = load_config(config_path)
            self.model = BiCodecTokenizer(
                model_dir = cache_dir,
                device = self.device
            )
            self.model.model.eval()

            logger.info(f"BiCodec tokenizer initialized on {self.device}")

            self.sample_rate = config["sample_rate"]
            self.volume_normalize = config["volume_normalize"]
        
        except Exception as e:
            logger.error(f"Error loading BiCodec: {e}")
            raise

    @property
    def codebook_size(self) -> int:
        """Size of the tokenizer's codebook."""
        return 8192
    
    @property
    def downsample_rate(self) -> int:
        """Temporal downsampling factor."""
        return 16000/50
    
    @property  
    def output_sample_rate(self) -> int:
        """Output sample rate after decoding."""
        return 16000

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to discrete tokens.

        Args:
            audio: Audio tensor (B, T) or (B, 1, T)

        Returns:
            global_tokens: Discrete token codes (B, Ng)
            semantic_tokens: Discrete token codes (B, Ns)
        """

        # Convert to numpy 
        wav = audio.squeeze().cpu().numpy()

        # Volume normalisation
        if self.volume_normalize:
            wav = audio_volume_normalize(wav)

        # Reference audio clip for speaker embedding
        ref_wav = self.model.get_ref_clip(wav)
        ref_wav = torch.from_numpy(ref_wav).unsqueeze(0).float()

        # Features for semantic token generation 
        feat = self.model.extract_wav2vec2_features(wav)
        batch = {
            "wav": torch.from_numpy(wav).unsqueeze(0).float().to(self.device),
            "ref_wav": ref_wav.to(self.device),
            "feat": feat.to(self.device),
        }

        # Generate semantic and global token
        semantic_tokens, global_tokens = self.model.model.tokenize(batch)

        return global_tokens, semantic_tokens
    
    def decode_tokens(self, semantic_tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens back to audio.

        Args:
            global_tokens: Token codes (B, Ng)
            semantic_tokens: Token codes (B, Ns)

        Returns:
            audio: Audio tensor (B, 1, T)
        """
        # Move to device
        
        semantic_tokens = semantic_tokens.to(self.device)
        
        try:
            global_tokens = getattr(semantic_tokens, "global_tokens")
        except Exception as e:
            logger.error(f"Error getting global token from attribute : {e}")
            raise 
        global_tokens = global_tokens.to(self.device)
        
        # Ensure correct shape (B, 1, Ng)
        if global_tokens.dim() == 2:
            global_tokens = global_tokens.unsqueeze(1)
        
        # Reconstruct wave 
        wav_rec = self.model.model.detokenize(semantic_tokens, global_tokens)

        # Ensure mono audio dimension (B, 1, T)
        if wav_rec.dim() == 2:
            wav_rec = wav_rec.unsqueeze(1)
        
        return wav_rec
    
    def encode(self,
            audio: Union[np.ndarray, torch.Tensor, str],
            sr: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Overrides the base encode method.
        Full encoding pipeline with preprocessing
        
        Returns:
            tokens: Encoded tokens
            info: Encoding information
        """
        # Preprocess
        audio_tensor = self.preprocess_audio(audio, sr)
        
        # Encode
        start_time = time.time()
        with torch.no_grad():
            global_tokens, semantic_tokens = self.encode_audio(audio_tensor)
        encode_time = time.time() - start_time
        
        # Info
        info = {
            "encode_time": encode_time,
            "input_shape": list(audio_tensor.shape),
            "global_token_shape": list(global_tokens.shape),
            "semantic_token_shape": list(semantic_tokens.shape),
            "num_tokens_semantic": semantic_tokens.numel(),
            "num_tokens_global": global_tokens.numel(),
            "num_tokens": semantic_tokens.numel() + global_tokens.numel(),
        }
        
        # Combine tokens
        semantic_tokens.global_tokens = global_tokens

        return semantic_tokens, info