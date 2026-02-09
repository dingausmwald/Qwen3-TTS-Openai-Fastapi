# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating TTS backend instances.
"""

import os
import logging
from typing import Optional

from .base import TTSBackend

logger = logging.getLogger(__name__)

# Global backend instance
_backend_instance: Optional[TTSBackend] = None


def get_backend() -> TTSBackend:
    """
    Get or create the global TTS backend instance.

    The backend is selected based on the TTS_BACKEND environment variable:
    - "optimized" (default): Optimized backend with torch.compile, CUDA graphs,
      model switching, voice prompt caching, and real-time streaming.
    - "official": Official Qwen3-TTS implementation (GPU/CPU auto-detect)
    - "vllm_omni": vLLM-Omni backend for optimized inference
    - "pytorch": CPU-optimized PyTorch backend
    - "openvino": Experimental OpenVINO backend for Intel CPUs

    Returns:
        TTSBackend instance
    """
    global _backend_instance

    if _backend_instance is not None:
        return _backend_instance

    backend_type = os.getenv("TTS_BACKEND", "optimized").lower()
    model_name = os.getenv("TTS_MODEL_NAME", os.getenv("TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))

    logger.info(f"Initializing TTS backend: {backend_type}")

    if backend_type == "optimized":
        from .optimized_backend import OptimizedQwen3TTSBackend
        _backend_instance = OptimizedQwen3TTSBackend()
        logger.info("Using optimized Qwen3-TTS backend")

    elif backend_type == "official":
        from .official_qwen3_tts import OfficialQwen3TTSBackend
        if model_name:
            _backend_instance = OfficialQwen3TTSBackend(model_name=model_name)
        else:
            _backend_instance = OfficialQwen3TTSBackend()
        logger.info(f"Using official backend: {_backend_instance.get_model_id()}")

    elif backend_type in ("vllm_omni", "vllm-omni", "vllm"):
        from .vllm_omni_qwen3_tts import VLLMOmniQwen3TTSBackend
        if model_name:
            _backend_instance = VLLMOmniQwen3TTSBackend(model_name=model_name)
        else:
            _backend_instance = VLLMOmniQwen3TTSBackend(
                model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            )
        logger.info(f"Using vLLM-Omni backend: {_backend_instance.get_model_id()}")

    elif backend_type == "pytorch":
        from .pytorch_backend import PyTorchCPUBackend
        device = os.getenv("TTS_DEVICE", "auto")
        dtype = os.getenv("TTS_DTYPE", "auto")
        attn = os.getenv("TTS_ATTN", "auto")
        cpu_threads = int(os.getenv("CPU_THREADS", "12"))
        cpu_interop = int(os.getenv("CPU_INTEROP", "2"))
        use_ipex = os.getenv("USE_IPEX", "false").lower() == "true"

        device_val = device if device != "auto" else "cpu"
        dtype_val = dtype if dtype != "auto" else "float32"
        attn_val = attn if attn != "auto" else "sdpa"

        _backend_instance = PyTorchCPUBackend(
            model_id=model_name,
            device=device_val,
            dtype=dtype_val,
            attn_implementation=attn_val,
            cpu_threads=cpu_threads,
            cpu_interop_threads=cpu_interop,
            use_ipex=use_ipex,
        )
        logger.info(f"Using CPU-optimized PyTorch backend: {_backend_instance.get_model_id()}")

    elif backend_type == "openvino":
        from .openvino_backend import OpenVINOBackend
        ov_device = os.getenv("OV_DEVICE", "CPU")
        ov_cache_dir = os.getenv("OV_CACHE_DIR", "./.ov_cache")
        ov_model_dir = os.getenv("OV_MODEL_DIR", "./.ov_models")

        _backend_instance = OpenVINOBackend(
            ov_model_dir=ov_model_dir,
            ov_device=ov_device,
            ov_cache_dir=ov_cache_dir,
        )
        logger.info("Using experimental OpenVINO backend")

    else:
        raise ValueError(
            f"Unknown TTS_BACKEND: {backend_type}. "
            f"Supported: 'optimized', 'official', 'vllm_omni', 'pytorch', 'openvino'"
        )

    return _backend_instance


async def initialize_backend(warmup: bool = False) -> TTSBackend:
    """
    Initialize the backend and optionally perform warmup.

    Args:
        warmup: Whether to run a warmup inference

    Returns:
        Initialized TTSBackend instance
    """
    backend = get_backend()

    # Optimized backend supports model_key parameter
    if hasattr(backend, 'initialize'):
        import inspect
        sig = inspect.signature(backend.initialize)
        if 'model_key' in sig.parameters:
            await backend.initialize()
        else:
            await backend.initialize()

    if warmup:
        warmup_enabled = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"
        if warmup_enabled:
            logger.info("Performing backend warmup...")
            try:
                await backend.generate_speech(
                    text="Hello, this is a warmup test.",
                    voice="Vivian",
                    language="English",
                )
                logger.info("Backend warmup completed")
            except Exception as e:
                logger.warning(f"Backend warmup failed (non-critical): {e}")

    return backend


def reset_backend() -> None:
    """Reset the global backend instance (useful for testing)."""
    global _backend_instance
    _backend_instance = None
