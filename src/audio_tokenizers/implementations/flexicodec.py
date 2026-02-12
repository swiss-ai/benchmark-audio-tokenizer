import os 
import sys
import time
import logging

import torch
import torchaudio
import numpy as np

flexicodec_path = os.path.join(os.path.dirname(__file__), '..','..','repos', 'FlexiCodec')
sys.path.insert(0, flexicodec_path)

from ..base import BaseAudioTokenizer

logger = logging.getLogger(__name__)

class FlexiCodecWrapper(BaseAudioTokenizer):
    name = "flexicodec"

    def _load_model(self):
        cache_dir = "/capstor/store/cscs/swissai/infra01/MLLM/flexicodec"
        os.makedirs(cache_dir, exist_ok = True)

        from huggingface_hub import snapshot_download, hf_hub_download
        sensevoice_small_path = snapshot_download(
                                repo_id='FunAudioLLM/SenseVoiceSmall', 
                                cache_dir=cache_dir,
                                local_dir=cache_dir
                                )
        
        req_file = os.path.join(sensevoice_small_path, "requirements.txt")
        if os.path.exists(req_file):
            os.remove(req_file)

        config_path = hf_hub_download(
                    repo_id ='jiaqili3/flexicodec', 
                    filename='12hz_v1_half_config.yaml',
                    cache_dir =cache_dir,
                    local_dir =cache_dir)
        ckpt_path = hf_hub_download(
                    repo_id='jiaqili3/flexicodec',
                    filename='12hz_v1_half.safetensors',
                    cache_dir =cache_dir,
                    local_dir =cache_dir)
        import yaml
        with open(config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        model_config['model']['semantic_model_path'] = sensevoice_small_path
        model_config['model']['semantic_model_type'] = 'sensevoice'

        def build_codec_model(config):
            from pathlib import Path
            import copy
            from flexicodec.modeling_flexicodec import FlexiCodec
            codec_model_config = copy.deepcopy(config)
            codec_model = FlexiCodec(
                **codec_model_config
            )
            return codec_model
        
        model = build_codec_model(model_config['model'])
        if ckpt_path.endswith('.safetensors'):
            import safetensors.torch
            model.load_state_dict(safetensors.torch.load_file(ckpt_path), strict=False)
        else:
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        from flexicodec.feature_extractors import FBankGen
        feature_extractor = FBankGen(sr=16000)
        self.model_dict = {"model": model, "feature_extractor": feature_extractor, "type": "sensevoice"}
    
    @property  
    def output_sample_rate(self) -> int:
        """Output sample rate after decoding."""
        return 16000  
    
    @output_sample_rate.setter
    def output_sample_rate(self, value: int):
        """Setter for output_sample_rate (for consistency)."""
        if value != 16000:
            logger.warning(f"Fixed at 16000Hz")
    
    @property
    def sample_rate(self) -> int:
        """Input sample rate for the tokenizer."""
        return 16000  # WavTokenizer uses 24kHz 
    
    @sample_rate.setter
    def sample_rate(self, value: int):
        """Setter for sample_rate (required by base class)."""
        if value != 16000:
            logger.warning(f"Fixed at 16000Hz")

    @property
    def downsample_rate(self) -> int:
        """Downsampling rate from audio samples to tokens."""
        return 16000/12.5  # 16000 / <12.5 = >1280 

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return 32768  # FSQ Codebook size for Semantic tokens


    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # Remove channel dimension
        
        audio = audio.to(self.device)

        from flexicodec.infer import encode_flexicodec
        with torch.no_grad():
            encoded_output = encode_flexicodec(audio, self.model_dict, self.sample_rate, num_quantizers=8, merging_threshold=0.91)
        
       
        sep_idx = encoded_output['semantic_codes'].shape[1]  # (B, 1, N)
        tokens = torch.cat([encoded_output['semantic_codes'], encoded_output['acoustic_codes']], dim=1) # (B, C, N)
        tokens.token_lengths = encoded_output['token_lengths']
        tokens.sep = sep_idx

        return tokens
        
    def decode_tokens(self, tokens, **kwargs) -> torch.Tensor:
        tokens = tokens.to(self.device)

        semantic_codes = tokens[:, :tokens.sep, :]
        acoustic_codes = tokens[:, tokens.sep:, :]
        token_lengths = tokens.token_lengths

        reconstructed_audio = self.model_dict['model'].decode_from_codes(
                        semantic_codes=semantic_codes,
                        acoustic_codes=acoustic_codes,
                        token_lengths=token_lengths,
                        )
        
        return reconstructed_audio
        
        

        
