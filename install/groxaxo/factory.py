# coding=utf-8
"""
Factory for creating TTS backend instances.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global backend instance
_backend_instance = None


def get_backend():
    """Get or create the global TTS backend instance."""
    global _backend_instance

    if _backend_instance is not None:
        return _backend_instance

    backend_type = os.getenv("TTS_BACKEND", "optimized").lower()

    if backend_type == "optimized":
        from .optimized_backend import OptimizedQwen3TTSBackend
        _backend_instance = OptimizedQwen3TTSBackend()
        logger.info("Using optimized Qwen3-TTS backend")

    elif backend_type == "official":
        from .official_qwen3_tts import OfficialQwen3TTSBackend
        model_name = os.getenv("TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        _backend_instance = OfficialQwen3TTSBackend(model_name=model_name)
        logger.info(f"Using official backend: {model_name}")

    else:
        raise ValueError(f"Unknown TTS_BACKEND: {backend_type}")

    return _backend_instance


async def initialize_backend(model_key: Optional[str] = None, warmup: bool = False):
    """Initialize the backend."""
    backend = get_backend()

    # Optimized backend supports model_key parameter
    if hasattr(backend, 'initialize'):
        import inspect
        sig = inspect.signature(backend.initialize)
        if 'model_key' in sig.parameters:
            await backend.initialize(model_key=model_key)
        else:
            await backend.initialize()

    if warmup:
        logger.info("Running warmup...")
        try:
            await backend.generate_speech("Warmup.", "Eric", "English")
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    return backend


def reset_backend():
    """Reset the backend instance."""
    global _backend_instance
    _backend_instance = None
