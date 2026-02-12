# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible router for text-to-speech API.
Implements endpoints compatible with OpenAI's TTS API specification.
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..structures.schemas import (
    OpenAISpeechRequest,
    ModelInfo,
    VoiceInfo,
    VoiceCloneRequest,
    VoiceCloneCapabilities,
)
from ..services.text_processing import normalize_text
from ..services.audio_encoding import encode_audio, get_content_type, DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)

# GPU lock: serializes TTS generation to prevent GPU contention
# (pattern from groxaxo vllm_omni backend)
_gpu_lock = asyncio.Lock()

# Voice library directory (same as VOICE_LIBRARY_DIR in start_server.sh)
VOICE_LIBRARY_DIR = Path(os.environ.get("VOICE_LIBRARY_DIR", "./voice_library")).resolve()

# Cache for reference audio reads (profile_name -> (audio, sr))
_ref_audio_cache = {}


def _load_voice_profile(name_or_id: str) -> dict:
    """Load a voice profile by name or profile_id from the voice library.

    Returns dict with keys: ref_audio_path, ref_text, x_vector_only_mode, language.
    Raises ValueError if not found.
    """
    profiles_dir = VOICE_LIBRARY_DIR / "profiles"
    if not profiles_dir.exists():
        raise ValueError(f"Voice library not found: {profiles_dir}")

    # Search all profiles
    for child in profiles_dir.iterdir():
        if not child.is_dir():
            continue
        meta_file = child / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Match by profile_id or name (case-insensitive)
        if meta.get("profile_id") == name_or_id or \
           meta.get("name", "").lower() == name_or_id.lower():
            # Found it -- build result
            ref_filename = meta.get("ref_audio_filename", "")
            if not ref_filename:
                raise ValueError(f"Profile '{name_or_id}' has no reference audio")
            ref_path = child / ref_filename
            if not ref_path.exists():
                raise ValueError(f"Reference audio missing: {ref_path}")
            return {
                "ref_audio_path": str(ref_path),
                "ref_text": meta.get("ref_text", ""),
                "x_vector_only_mode": meta.get("x_vector_only_mode", False),
                "language": meta.get("language", "Auto"),
                "name": meta.get("name", name_or_id),
            }

    raise ValueError(f"Voice profile not found: '{name_or_id}'")


# Language code to language name mapping
LANGUAGE_CODE_MAPPING = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

# Available models (including language-specific variants)
AVAILABLE_MODELS = [
    ModelInfo(
        id="qwen3-tts",
        object="model",
        created=1737734400,  # 2025-01-24
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1-hd",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
]

# Add language-specific model variants
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    AVAILABLE_MODELS.extend([
        ModelInfo(
            id=f"tts-1-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
        ModelInfo(
            id=f"tts-1-hd-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
    ])

# Model name mapping (OpenAI -> internal)
MODEL_MAPPING = {
    "tts-1": "qwen3-tts",
    "tts-1-hd": "qwen3-tts",
    "qwen3-tts": "qwen3-tts",
}

# Add language-specific model mappings
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    MODEL_MAPPING[f"tts-1-{lang_code}"] = "qwen3-tts"
    MODEL_MAPPING[f"tts-1-hd-{lang_code}"] = "qwen3-tts"

# OpenAI voice mapping to Qwen voices
# Must map to voices that actually exist in the model:
# aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
VOICE_MAPPING = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Serena",
    "nova": "Aiden",
    "onyx": "Eric",
    "shimmer": "Dylan",
}


def extract_language_from_model(model_name: str) -> Optional[str]:
    """Extract language from model name if it has a language suffix."""
    for lang_code, lang_name in LANGUAGE_CODE_MAPPING.items():
        suffix = f"-{lang_code}"
        if model_name.endswith(suffix):
            if model_name == f"tts-1{suffix}" or model_name == f"tts-1-hd{suffix}":
                return lang_name
    return None


async def get_tts_backend():
    """Get the TTS backend instance, initializing if needed."""
    from ..backends import get_backend, initialize_backend

    backend = get_backend()

    if not backend.is_ready():
        await initialize_backend()

    return backend


def get_voice_name(voice: str) -> str:
    """Map voice name to internal voice identifier."""
    if voice.lower() in VOICE_MAPPING:
        return VOICE_MAPPING[voice.lower()]
    return voice


async def generate_speech(
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Generate speech from text using the configured TTS backend."""
    backend = await get_tts_backend()
    voice_name = get_voice_name(voice)

    try:
        audio, sr = await backend.generate_speech(
            text=text,
            voice=voice_name,
            language=language,
            instruct=instruct,
            speed=speed,
        )
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Speech generation failed: {e}")


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
):
    """OpenAI-compatible endpoint for text-to-speech."""
    logger.info(f"TTS request: model={request.model}, voice={request.voice}, format={request.response_format}, len={len(request.input)}")

    if request.model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}. Supported: {list(MODEL_MAPPING.keys())}",
                "type": "invalid_request_error",
            },
        )

    try:
        normalized_text = normalize_text(request.input, request.normalization_options)

        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_input",
                    "message": "Input text is empty after normalization",
                    "type": "invalid_request_error",
                },
            )

        model_language = extract_language_from_model(request.model)
        language = model_language if model_language else (request.language or "Auto")

        # Voice profile: voice name starts with "clone:" -> load from voice library
        if request.voice.lower().startswith("clone:"):
            profile_name = request.voice[6:].strip()
            if not profile_name:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_voice",
                        "message": "clone: prefix requires a profile name, e.g. 'clone:MyVoice'",
                        "type": "invalid_request_error",
                    },
                )
            try:
                profile = _load_voice_profile(profile_name)
            except ValueError as e:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "profile_not_found",
                        "message": str(e),
                        "type": "invalid_request_error",
                    },
                )

            backend = await get_tts_backend()

            # Cache reference audio reads per profile
            ref_audio_path = profile["ref_audio_path"]
            if profile_name in _ref_audio_cache:
                ref_audio, ref_sr = _ref_audio_cache[profile_name]
            else:
                ref_audio, ref_sr = sf.read(ref_audio_path)
                if len(ref_audio.shape) > 1:
                    ref_audio = ref_audio.mean(axis=1)
                ref_audio = ref_audio.astype(np.float32)
                _ref_audio_cache[profile_name] = (ref_audio, ref_sr)
                logger.info(f"Reference audio cached: '{profile_name}'")

            clone_lang = language if language != "Auto" else profile["language"]
            logger.info(f"Voice profile '{profile['name']}': lang={clone_lang}, xvec_only={profile['x_vector_only_mode']}, stream={request.stream}")

            if request.stream:
                # True streaming: yield PCM chunks as model generates them
                fmt = request.response_format
                if fmt == "wav":
                    fmt = "pcm"
                content_type = get_content_type(fmt)

                async def clone_audio_stream():
                    gen_start = time.time()
                    first_chunk_time = None
                    total_samples = 0
                    chunk_count = 0
                    sample_rate = 24000
                    async for pcm_chunk, sr in backend.generate_voice_clone_streaming(
                        text=normalized_text,
                        ref_audio=ref_audio,
                        ref_audio_sr=ref_sr,
                        ref_text=profile["ref_text"] or None,
                        language=clone_lang,
                        x_vector_only_mode=profile["x_vector_only_mode"],
                        cache_key=profile_name,
                    ):
                        if pcm_chunk is not None and len(pcm_chunk) > 0:
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - gen_start
                            total_samples += len(pcm_chunk)
                            sample_rate = sr
                            chunk_count += 1
                            yield encode_audio(pcm_chunk, fmt, sr)
                            await asyncio.sleep(0)
                    gen_time = time.time() - gen_start
                    audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
                    rtf = gen_time / audio_dur if audio_dur > 0 else 0
                    logger.info(f"Voice clone stream: First-Byte={first_chunk_time:.2f}s Gesamt={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x Chunks={chunk_count}")

                return StreamingResponse(
                    clone_audio_stream(),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"inline; filename=speech.{fmt}",
                        "Cache-Control": "no-cache",
                    },
                )
            else:
                # Non-streaming: generate all audio, then send (better RTF)
                gen_start = time.time()
                audio, sample_rate = await backend.generate_voice_clone(
                    text=normalized_text,
                    ref_audio=ref_audio,
                    ref_audio_sr=ref_sr,
                    ref_text=profile["ref_text"] or None,
                    language=clone_lang,
                    x_vector_only_mode=profile["x_vector_only_mode"],
                    speed=request.speed,
                    cache_key=profile_name,
                )
                gen_time = time.time() - gen_start

                audio_dur = len(audio) / sample_rate if sample_rate > 0 else 0
                rtf = gen_time / audio_dur if audio_dur > 0 else 0
                logger.info(f"Voice clone: Gen={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x")

                fmt = request.response_format
                if fmt == "wav":
                    fmt = "mp3"
                audio_bytes = encode_audio(audio, fmt, sample_rate)
                content_type = get_content_type(fmt)

                async def send_audio():
                    for i in range(0, len(audio_bytes), 4096):
                        yield audio_bytes[i:i + 4096]

                return StreamingResponse(
                    send_audio(),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"inline; filename=speech.{fmt}",
                        "Cache-Control": "no-cache",
                    },
                )

        # Voice cloning: task_type=Base with ref_audio (Voice Studio)
        if request.task_type == "Base" and request.ref_audio:
            backend = await get_tts_backend()

            ref_audio_data = request.ref_audio
            if ref_audio_data.startswith("data:"):
                ref_audio_data = ref_audio_data.split(",", 1)[1]
            audio_bytes_raw = base64.b64decode(ref_audio_data)
            audio_buffer = io.BytesIO(audio_bytes_raw)
            ref_audio, ref_sr = sf.read(audio_buffer)
            if len(ref_audio.shape) > 1:
                ref_audio = ref_audio.mean(axis=1)
            ref_audio = ref_audio.astype(np.float32)

            logger.info(f"Voice clone: lang={language}, ref_text={request.ref_text is not None}, xvec_only={request.x_vector_only_mode}")

            gen_start = time.time()
            audio, sample_rate = await backend.generate_voice_clone(
                text=normalized_text,
                ref_audio=ref_audio,
                ref_audio_sr=ref_sr,
                ref_text=request.ref_text,
                language=language,
                x_vector_only_mode=request.x_vector_only_mode or False,
                speed=request.speed,
            )
            gen_time = time.time() - gen_start

            audio_dur = len(audio) / sample_rate if sample_rate > 0 else 0
            rtf = gen_time / audio_dur if audio_dur > 0 else 0
            logger.info(f"Voice clone: Gen={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x")

            audio_bytes = encode_audio(audio, request.response_format, sample_rate)
            content_type = get_content_type(request.response_format)

            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",
                },
            )

        if request.stream:
            backend = await get_tts_backend()
            voice_name = get_voice_name(request.voice)
            fmt = request.response_format
            if fmt == "wav":
                fmt = "pcm"
            content_type = get_content_type(fmt)

            async def audio_stream():
                gen_start = time.time()
                first_chunk_time = None
                total_samples = 0
                chunk_count = 0
                sample_rate = 24000
                async for pcm_chunk, sr in backend.generate_speech_streaming(
                    text=normalized_text,
                    voice=voice_name,
                    language=language,
                    instruct=request.instruct,
                    model=request.model,
                ):
                    if pcm_chunk is not None and len(pcm_chunk) > 0:
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - gen_start
                        total_samples += len(pcm_chunk)
                        sample_rate = sr
                        chunk_count += 1
                        yield encode_audio(pcm_chunk, fmt, sr)
                        await asyncio.sleep(0)
                gen_time = time.time() - gen_start
                audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
                rtf = gen_time / audio_dur if audio_dur > 0 else 0
                logger.info(f"TTS stream: First-Byte={first_chunk_time:.2f}s Gesamt={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x Chunks={chunk_count}")

            return StreamingResponse(
                audio_stream(),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{fmt}",
                    "Cache-Control": "no-cache",
                },
            )
        else:
            # Non-streaming: generate in thread pool to keep event loop free
            # GPU lock serializes access (pattern from groxaxo vllm_omni backend)
            async with _gpu_lock:
                gen_start = time.time()
                loop = asyncio.get_event_loop()
                audio, sample_rate = await loop.run_in_executor(
                    None,
                    lambda: asyncio.run(generate_speech(
                        text=normalized_text,
                        voice=request.voice,
                        language=language,
                        instruct=request.instruct,
                        speed=request.speed,
                    ))
                )
                gen_time = time.time() - gen_start

            audio_dur = len(audio) / sample_rate if sample_rate > 0 else 0
            rtf = gen_time / audio_dur if audio_dur > 0 else 0
            logger.info(f"TTS: Gen={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x")

            audio_bytes = encode_audio(audio, request.response_format, sample_rate)
            content_type = get_content_type(request.response_format)

            async def send_audio():
                for i in range(0, len(audio_bytes), 4096):
                    yield audio_bytes[i:i + 4096]

            return StreamingResponse(
                send_audio(),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"inline; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"TTS request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.get("/models")
async def list_models():
    """List all available TTS models."""
    return {
        "object": "list",
        "data": [model.model_dump() for model in AVAILABLE_MODELS],
    }


@router.get("/audio/models")
async def list_audio_models():
    """List TTS models in OpenWebUI-compatible format."""
    return {
        "models": [model.model_dump() for model in AVAILABLE_MODELS],
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model.model_dump()

    raise HTTPException(
        status_code=404,
        detail={
            "error": "model_not_found",
            "message": f"Model '{model_id}' not found",
            "type": "invalid_request_error",
        },
    )


@router.get("/audio/voices")
@router.get("/voices")
async def list_voices():
    """List all available voices for text-to-speech."""
    openai_voices = [
        VoiceInfo(id="alloy", name="Alloy", description="OpenAI-compatible voice (maps to Vivian)"),
        VoiceInfo(id="echo", name="Echo", description="OpenAI-compatible voice (maps to Ryan)"),
        VoiceInfo(id="fable", name="Fable", description="OpenAI-compatible voice (maps to Serena)"),
        VoiceInfo(id="nova", name="Nova", description="OpenAI-compatible voice (maps to Aiden)"),
        VoiceInfo(id="onyx", name="Onyx", description="OpenAI-compatible voice (maps to Eric)"),
        VoiceInfo(id="shimmer", name="Shimmer", description="OpenAI-compatible voice (maps to Dylan)"),
    ]

    default_languages = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Russian", "Portuguese", "Italian"]

    try:
        backend = await get_tts_backend()
        speakers = backend.get_supported_voices()
        languages = backend.get_supported_languages()

        if speakers:
            voices = []
            for speaker in speakers:
                voice_info = VoiceInfo(
                    id=speaker,
                    name=speaker,
                    language=languages[0] if languages else "Auto",
                    description=f"Qwen3-TTS voice: {speaker}",
                )
                voices.append(voice_info.model_dump())
        else:
            voices = []

        # Add clone profiles from voice library
        clone_voices = []
        profiles_dir = VOICE_LIBRARY_DIR / "profiles"
        if profiles_dir.exists():
            for child in sorted(profiles_dir.iterdir()):
                meta_file = child / "meta.json"
                if not meta_file.exists():
                    continue
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if meta.get("task_type") == "Base" and meta.get("ref_audio_filename"):
                        clone_id = f"clone:{meta['name']}"
                        clone_voices.append(VoiceInfo(
                            id=clone_id,
                            name=clone_id,
                            description=f"Cloned voice: {meta['name']}",
                        ).model_dump())
                except Exception:
                    pass

        return {
            "voices": voices + clone_voices + [v.model_dump() for v in openai_voices],
            "languages": languages if languages else default_languages,
        }

    except Exception as e:
        logger.warning(f"Could not get voices from backend: {e}")
        return {
            "voices": [v.model_dump() for v in openai_voices],
            "languages": default_languages,
        }


@router.get("/audio/voice-clone/capabilities")
async def get_voice_clone_capabilities():
    """Get voice cloning capabilities of the current backend."""
    try:
        backend = await get_tts_backend()
        supports_cloning = backend.supports_voice_cloning()
        model_type = backend.get_model_type() if hasattr(backend, 'get_model_type') else "unknown"

        return VoiceCloneCapabilities(
            supported=supports_cloning,
            model_type=model_type,
            icl_mode_available=supports_cloning,
            x_vector_mode_available=supports_cloning,
        )
    except Exception as e:
        logger.warning(f"Could not get voice clone capabilities: {e}")
        return VoiceCloneCapabilities(
            supported=False,
            model_type="unknown",
            icl_mode_available=False,
            x_vector_mode_available=False,
        )


@router.post("/audio/voice-clone")
async def create_voice_clone(
    request: VoiceCloneRequest,
    client_request: Request,
):
    """Clone a voice from reference audio and generate speech."""
    try:
        backend = await get_tts_backend()

        if not backend.supports_voice_cloning():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "voice_cloning_not_supported",
                    "message": "Voice cloning requires the Base model.",
                    "type": "invalid_request_error",
                },
            )

        if not request.x_vector_only_mode and not request.ref_text:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_ref_text",
                    "message": "ICL mode requires ref_text. Either provide ref_text or set x_vector_only_mode=True.",
                    "type": "invalid_request_error",
                },
            )

        try:
            audio_bytes = base64.b64decode(request.ref_audio)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_audio", "message": f"Failed to decode base64 audio: {e}", "type": "invalid_request_error"},
            )

        try:
            audio_buffer = io.BytesIO(audio_bytes)
            ref_audio, ref_sr = sf.read(audio_buffer)
            if len(ref_audio.shape) > 1:
                ref_audio = ref_audio.mean(axis=1)
            ref_audio = ref_audio.astype(np.float32)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "audio_processing_error", "message": f"Failed to process reference audio: {e}", "type": "invalid_request_error"},
            )

        normalized_text = normalize_text(request.input, request.normalization_options)
        if not normalized_text.strip():
            raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "Input text is empty after normalization", "type": "invalid_request_error"})

        audio, sample_rate = await backend.generate_voice_clone(
            text=normalized_text, ref_audio=ref_audio, ref_audio_sr=ref_sr,
            ref_text=request.ref_text, language=request.language or "Auto",
            x_vector_only_mode=request.x_vector_only_mode, speed=request.speed,
        )

        audio_bytes = encode_audio(audio, request.response_format, sample_rate)
        content_type = get_content_type(request.response_format)

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename=voice_clone.{request.response_format}", "Cache-Control": "no-cache"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail={"error": "processing_error", "message": str(e), "type": "server_error"})
