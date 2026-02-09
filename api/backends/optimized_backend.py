# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Optimized Qwen3-TTS backend with torch.compile, CUDA graphs,
dynamic model switching, voice prompt caching, and real-time streaming.

Requires dffdeeq/Qwen3-TTS-streaming as the inference library.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("TTS_CONFIG", Path.home() / "qwen3-tts" / "config.yaml"))

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}

# Model mapping: OpenAI model name -> our model key
MODEL_MAPPING = {
    "tts-1": "0.6B-CustomVoice",
    "tts-1-hd": "1.7B-CustomVoice",
    "qwen3-tts": "0.6B-CustomVoice",
    "qwen3-tts-hd": "1.7B-CustomVoice",
}

class OptimizedQwen3TTSBackend:
    """Optimized backend with dynamic model switching."""

    def __init__(self):
        self.config = load_config()
        self.models = {}  # Cache loaded models
        self.current_model_key = None
        self.model = None
        self._ready = False
        self.device = None
        self.dtype = None
        self._voice_prompt_cache = {}  # cache_key -> VoiceClonePromptItem list

        # Set environment variables from config
        for key, value in self.config.get("env", {}).items():
            os.environ[key] = str(value)
            logger.info(f"Set {key}={value}")

    def _get_model_key_for_request(self, openai_model: str) -> str:
        """Map OpenAI model name to our model key."""
        if openai_model in MODEL_MAPPING:
            return MODEL_MAPPING[openai_model]
        base_model = openai_model.rsplit('-', 1)[0] if '-' in openai_model else openai_model
        if base_model in MODEL_MAPPING:
            return MODEL_MAPPING[base_model]
        return self.config.get("default_model", "0.6B-CustomVoice")

    def get_available_models(self) -> List[str]:
        return list(self.config.get("models", {}).keys())

    def get_model_info(self, model_key: str) -> dict:
        return self.config.get("models", {}).get(model_key, {})

    def get_current_model_key(self) -> Optional[str]:
        return self.current_model_key

    async def _ensure_model_loaded(self, model_key: str) -> None:
        """Load model if not already loaded."""
        import torch

        if self.current_model_key == model_key and self.model is not None:
            return

        model_info = self.config.get("models", {}).get(model_key, {})
        if not model_info:
            raise ValueError(f"Unknown model: {model_key}")

        hf_id = model_info["hf_id"]

        # Unload previous model
        if self.model is not None:
            logger.info(f"Unloading {self.current_model_key}...")
            if self._voice_prompt_cache:
                logger.info(f"Clearing voice prompt cache ({len(self._voice_prompt_cache)} entries)")
                self._voice_prompt_cache.clear()
            del self.model
            self.model = None
            torch.cuda.empty_cache()

        logger.info(f"Loading {model_key} ({hf_id})...")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        from qwen_tts import Qwen3TTSModel
        torch.set_float32_matmul_precision('high')

        opt = self.config.get("optimization", {})
        attn_impl = opt.get("attention", "flash_attention_2")

        try:
            self.model = Qwen3TTSModel.from_pretrained(
                hf_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_impl,
            )
            logger.info(f"Loaded with {attn_impl}")
        except Exception as e:
            logger.warning(f"Failed with {attn_impl}: {e}, trying sdpa")
            self.model = Qwen3TTSModel.from_pretrained(
                hf_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation="sdpa",
            )

        # torch.compile + warmup
        if opt.get("use_compile", True) and self.device != "cpu":
            try:
                streaming_opts = opt.get("streaming", {})
                self.model.enable_streaming_optimizations(
                    decode_window_frames=streaming_opts.get("decode_window_frames", 80),
                    use_compile=True,
                    use_cuda_graphs=opt.get("use_cuda_graphs", False),
                    compile_mode=opt.get("compile_mode", "max-autotune"),
                    use_fast_codebook=opt.get("use_fast_codebook", True),
                    compile_codebook_predictor=opt.get("compile_codebook_predictor", True),
                )
                logger.info(f"torch.compile enabled: mode={opt.get('compile_mode', 'max-autotune')}, fast_codebook={opt.get('use_fast_codebook', True)}")

                # Mandatory warmup — triggers actual kernel compilation
                model_type = model_info.get("type", "customvoice")

                if model_type == "base":
                    logger.info("Warmup: compiling non-streaming kernels (base/voice-clone)...")
                    dummy_audio = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)
                    self.model.generate_voice_clone(
                        text="Warmup sentence for voice clone compilation.",
                        language="English",
                        ref_audio=(dummy_audio, 24000),
                        x_vector_only_mode=True,
                    )
                    logger.info("Non-streaming warmup complete (base)")

                    logger.info("Warmup: compiling streaming kernels (base/voice-clone)...")
                    for chunk, sr_out in self.model.stream_generate_voice_clone(
                        text="Streaming warmup for voice clone.",
                        language="English",
                        ref_audio=(dummy_audio, 24000),
                        x_vector_only_mode=True,
                        emit_every_frames=streaming_opts.get("emit_every_frames", 4),
                        decode_window_frames=streaming_opts.get("decode_window_frames", 80),
                    ):
                        pass
                    logger.info("Streaming warmup complete (base)")
                else:
                    logger.info("Warmup: compiling non-streaming kernels (custom voice)...")
                    self.model.generate_custom_voice(
                        text="This is a warmup sentence to trigger torch compile kernel compilation.",
                        language="English",
                        speaker="Eric",
                    )
                    logger.info("Non-streaming warmup complete")

                    logger.info("Warmup: compiling streaming kernels (custom voice)...")
                    for chunk, sr_out in self.model.stream_generate_custom_voice(
                        text="Streaming warmup sentence for compiling the streaming code path.",
                        speaker="Eric",
                        language="English",
                        emit_every_frames=streaming_opts.get("emit_every_frames", 4),
                        decode_window_frames=streaming_opts.get("decode_window_frames", 80),
                    ):
                        pass
                    logger.info("Streaming warmup complete")
            except Exception as e:
                logger.warning(f"Could not enable optimizations: {e}")

        self.current_model_key = model_key
        self._ready = True
        logger.info(f"Model {model_key} ready on {self.device}")

    async def initialize(self, model_key: Optional[str] = None) -> None:
        """Initialize with default or specified model."""
        if model_key is None:
            model_key = self.config.get("default_model", "0.6B-CustomVoice")
        await self._ensure_model_loaded(model_key)

    async def switch_model(self, model_key: str) -> None:
        """Switch to a different model."""
        await self._ensure_model_loaded(model_key)

    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
        model: str = "tts-1",
    ) -> Tuple[np.ndarray, int]:
        """Generate speech, switching model if needed."""
        model_key = self._get_model_key_for_request(model)
        await self._ensure_model_loaded(model_key)

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=voice,
            instruct=instruct,
        )

        audio = wavs[0]
        if speed != 1.0:
            try:
                import librosa
                audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
            except ImportError:
                pass
        return audio, sr


    async def generate_speech_streaming(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: str = None,
        speed: float = 1.0,
        model: str = "tts-1",
    ):
        """Real token-by-token streaming via dffdeeq stream_generate_custom_voice.

        Yields (pcm_chunk, sample_rate) tuples as audio is generated.
        """
        model_key = self._get_model_key_for_request(model)
        await self._ensure_model_loaded(model_key)

        streaming_opts = self.config.get("optimization", {}).get("streaming", {})
        decode_window_frames = streaming_opts.get("decode_window_frames", 80)
        emit_every_frames = streaming_opts.get("emit_every_frames", 4)

        voice_name = voice
        openai_mapping = {
            "alloy": "Vivian", "echo": "Ryan", "fable": "Sophia",
            "nova": "Isabella", "onyx": "Evan", "shimmer": "Lily",
        }
        if voice.lower() in openai_mapping:
            voice_name = openai_mapping[voice.lower()]

        for chunk, sr in self.model.stream_generate_custom_voice(
            text=text,
            speaker=voice_name,
            language=language,
            instruct=instruct,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
        ):
            yield chunk, sr


    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
        cache_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Voice cloning requires Base model."""
        await self._ensure_model_loaded("0.6B-Base")

        import time as _time

        # Use cached voice prompt if available, otherwise compute and cache
        if cache_key and cache_key in self._voice_prompt_cache:
            prompt_items = self._voice_prompt_cache[cache_key]
            t0 = _time.time()
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt_items,
            )
            logger.info(f"Voice clone (cached): generate={_time.time()-t0:.3f}s")
        else:
            if cache_key:
                t0 = _time.time()
                prompt_items = self.model.create_voice_clone_prompt(
                    ref_audio=(ref_audio, ref_audio_sr),
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                )
                t_prompt = _time.time() - t0
                self._voice_prompt_cache[cache_key] = prompt_items
                logger.info(f"Voice prompt cached: '{cache_key}' (prompt_build={t_prompt:.3f}s)")
                t0 = _time.time()
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt_items,
                )
                logger.info(f"Voice clone (first): prompt_build={t_prompt:.3f}s generate={_time.time()-t0:.3f}s")
            else:
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    ref_audio=(ref_audio, ref_audio_sr),
                    ref_text=ref_text,
                    language=language,
                    x_vector_only_mode=x_vector_only_mode,
                )

        audio = wavs[0]
        if speed != 1.0:
            try:
                import librosa
                audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
            except ImportError:
                pass
        return audio, sr

    async def generate_voice_clone_streaming(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        cache_key: Optional[str] = None,
    ):
        """Streaming voice cloning via dffdeeq stream_generate_voice_clone.

        Yields (pcm_chunk, sample_rate) tuples as audio is generated.
        """
        await self._ensure_model_loaded("0.6B-Base")

        import time as _time

        streaming_opts = self.config.get("optimization", {}).get("streaming", {})
        decode_window_frames = streaming_opts.get("decode_window_frames", 80)
        emit_every_frames = streaming_opts.get("emit_every_frames", 4)

        # Build or retrieve cached voice clone prompt
        if cache_key and cache_key in self._voice_prompt_cache:
            prompt_items = self._voice_prompt_cache[cache_key]
        else:
            t0 = _time.time()
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=(ref_audio, ref_audio_sr),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            t_prompt = _time.time() - t0
            if cache_key:
                self._voice_prompt_cache[cache_key] = prompt_items
                logger.info(f"Voice prompt cached: '{cache_key}' (prompt_build={t_prompt:.3f}s)")
            else:
                logger.info(f"Voice prompt built (no cache): {t_prompt:.3f}s")

        for chunk, sr in self.model.stream_generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt_items,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
        ):
            yield chunk, sr

    def get_backend_name(self) -> str:
        return "optimized"

    def get_model_id(self) -> str:
        if self.current_model_key:
            info = self.config.get("models", {}).get(self.current_model_key, {})
            return info.get("hf_id", "unknown")
        return "not-loaded"

    def get_supported_voices(self) -> List[str]:
        return [v["name"] for v in self.config.get("voices", [])]

    def get_supported_languages(self) -> List[str]:
        return ["English", "Chinese", "Japanese", "Korean", "German",
                "French", "Spanish", "Russian", "Portuguese", "Italian"]

    def is_ready(self) -> bool:
        return self._ready

    def supports_voice_cloning(self) -> bool:
        return True

    def get_model_type(self) -> str:
        if not self.current_model_key:
            return "unknown"
        return self.config.get("models", {}).get(self.current_model_key, {}).get("type", "unknown")

    def get_device_info(self) -> Dict[str, Any]:
        import torch
        info = {"device": str(self.device) if self.device else "unknown", "gpu_available": False}
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_total"] = f"{props.total_memory / 1024**3:.1f} GB"
        return info
