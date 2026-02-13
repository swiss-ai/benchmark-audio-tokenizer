"""
Audio tokenizer implementations.

Each tokenizer here is automatically registered when imported.
Uses lazy imports to only load dependencies for requested tokenizers.
"""

import logging
import importlib
logger = logging.getLogger(__name__)

# Tokenizer import mappings: name -> (module, classes)
_TOKENIZER_IMPORTS = {
    'neucodec': ('neucodec', ['NeuCodecTokenizer', 'DistilledNeuCodecTokenizer']),
    'tadicodec': ('tadicodec', ['TaDiCodecTokenizer']),
    'xcodec2': ('xcodec2', ['XCodec2Tokenizer']),
    'cosyvoice2': ('cosyvoice2', ['CosyVoice2Tokenizer']),
    'glm4voice': ('glm4voice', ['GLM4VoiceTokenizer']),
    'wavtokenizer': ('wavtokenizer', ['WavTokenizerWrapper']),
    'stepaudioeditx': ('stepaudioeditx', ['StepAudioEditXWrapper']),
    'mimoaudio': ('mimoaudio', ['MiMoAudioTokenizer']),
    'varstok': ('varstok', ['VARSTokWrapper'])
}

# Try to import all tokenizers, but don't fail if dependencies are missing
_available_tokenizers = []
for tokenizer_key, (module_name, class_names) in _TOKENIZER_IMPORTS.items():
    try:
        module = importlib.import_module(f'.{module_name}', package='audio_tokenizers.implementations')
        for class_name in class_names:
            cls = getattr(module, class_name)
            globals()[class_name] = cls
            _available_tokenizers.append(class_name)
    except ImportError as e:
        logger.debug(f"Tokenizer {tokenizer_key} not available: {e}")
        # Set to None so we know it was attempted
        for class_name in class_names:
            globals()[class_name] = None

__all__ = _available_tokenizers
