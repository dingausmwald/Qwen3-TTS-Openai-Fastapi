# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Based on dffdeeq/Qwen3-TTS-streaming.
# Patch: added stream_generate_custom_voice() for CustomVoice model streaming.
import base64
import io
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from ..core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """
    ref_code: Optional[torch.Tensor]                 # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_spk_embedding: torch.Tensor                  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


class Qwen3TTSModel:
    """
    A HuggingFace-style wrapper for Qwen3 TTS models (CustomVoice/VoiceDesign/Base) that provides:
      - from_pretrained() initialization via AutoModel/AutoProcessor
      - generation APIs for:
          * CustomVoice: generate_custom_voice()
          * VoiceDesign: generate_voice_design()
          * Base: generate_voice_clone() + create_voice_clone_prompt()
      - consistent output: (wavs: List[np.ndarray], sample_rate: int)

    Notes:
      - This wrapper expects the underlying model class to be `Qwen3TTSForConditionalGeneration`
      - Language / speaker validation is done via model methods:
          model.get_supported_languages(), model.get_supported_speakers()
    """

    def __init__(self, model: Qwen3TTSForConditionalGeneration, processor, generate_defaults: Optional[Dict[str, Any]] = None):
        self.model = model
        self.processor = processor
        self.generate_defaults = generate_defaults or {}

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "Qwen3TTSModel":
        """
        Load a Qwen3 TTS model and its processor in HuggingFace `from_pretrained` style.

        This method:
          1) Loads config via AutoConfig (so your side can register model_type -> config/model).
          2) Loads the model via AutoModel.from_pretrained(...), forwarding `kwargs` unchanged.
          3) Loads the processor via AutoProcessor.from_pretrained(model_path).
          4) Loads optional `generate_config.json` from the model directory/repo snapshot if present.

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory of the model.
            **kwargs:
                Forwarded as-is into `AutoModel.from_pretrained(...)`.
                Typical examples: device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2".

        Returns:
            Qwen3TTSModel:
                Wrapper instance containing `model`, `processor`, and generation defaults.
        """
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration. "
            )

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True,)

        generate_defaults = model.generate_config
        return cls(model=model, processor=processor, generate_defaults=generate_defaults)

    def enable_streaming_optimizations(
        self,
        decode_window_frames: int = 80,
        use_compile: bool = True,
        use_cuda_graphs: bool = True,
        compile_mode: str = "reduce-overhead",
        use_fast_codebook: bool = False,  # Disabled: needs debugging, currently slower
        compile_codebook_predictor: bool = True,
    ):
        """
        Enable torch.compile and CUDA graphs optimizations for streaming decode.

        Significantly speeds up streaming generation by:
        1. Compiling the decoder with torch.compile (reduces Python overhead)
        2. Capturing CUDA graphs for fixed-size decode windows (eliminates GPU launch overhead)
        3. Fast codebook generation (bypasses HuggingFace generate() overhead)

        Call this method after loading the model, before starting streaming generation.

        Args:
            decode_window_frames: Fixed window size for streaming (must match the
                                  decode_window_frames parameter in stream_generate_voice_clone)
            use_compile: Apply torch.compile to the decoder (default True)
            use_cuda_graphs: Capture CUDA graphs for the fixed window size (default True)
            compile_mode: torch.compile mode - "reduce-overhead" (recommended for streaming),
                          "max-autotune", or "default"
            use_fast_codebook: Use fast codebook generation that bypasses HuggingFace's
                               generate() overhead (default True, ~2x faster per step)
            compile_codebook_predictor: Apply torch.compile to codebook predictor (experimental)

        Returns:
            self for method chaining

        Example:
            model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", ...)
            model.enable_streaming_optimizations(decode_window_frames=80)

            # Now streaming will be faster
            for chunk, sr in model.stream_generate_voice_clone(..., decode_window_frames=80):
                ...
        """
        self.model.enable_streaming_optimizations(
            decode_window_frames=decode_window_frames,
            use_compile=use_compile,
            use_cuda_graphs=use_cuda_graphs,
            compile_mode=compile_mode,
            use_fast_codebook=use_fast_codebook,
            compile_codebook_predictor=compile_codebook_predictor,
        )
        return self

    def _supported_languages_set(self) -> Optional[set]:
        langs = getattr(self.model, "get_supported_languages", None)
        if callable(langs):
            v = langs()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _supported_speakers_set(self) -> Optional[set]:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _validate_languages(self, languages: List[str]) -> None:
        supported = self._supported_languages_set()
        if supported is None:
            return
        bad = []
        for lang in languages:
            if lang is None:
                bad.append(lang)
                continue
            if str(lang).lower() not in supported:
                bad.append(lang)
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")

    def _validate_speakers(self, speakers: List[Optional[str]]) -> None:
        supported = self._supported_speakers_set()
        if supported is None:
            return
        bad = []
        for spk in speakers:
            if spk is None or spk == "":
                continue
            if str(spk).lower() not in supported:
                bad.append(spk)
        if bad:
            raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        if self._is_url(x):
            with urllib.request.urlopen(x) as resp:
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        elif self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        for i, a in enumerate(out):
            if a[0].ndim > 1:
                a[0] = np.mean(a[0], axis=-1).astype(np.float32)
                out[i] = (a[0], a[1])
        return out

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, texts: List[str]) -> List[torch.Tensor]:
        input_ids = []
        for text in texts:
            input = self.processor(text=text, return_tensors="pt", padding=True)
            input_id = input["input_ids"].to(self.device)
            input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
            input_ids.append(input_id)
        return input_ids

    def _merge_generate_kwargs(
        self,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        hard_defaults = dict(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=2048,
        )

        def pick(name: str, user_val: Any) -> Any:
            if user_val is not None:
                return user_val
            if name in self.generate_defaults:
                return self.generate_defaults[name]
            return hard_defaults[name]

        merged = dict(kwargs)
        merged.update(
            do_sample=pick("do_sample", do_sample),
            top_k=pick("top_k", top_k),
            top_p=pick("top_p", top_p),
            temperature=pick("temperature", temperature),
            repetition_penalty=pick("repetition_penalty", repetition_penalty),
            subtalker_dosample=pick("subtalker_dosample", subtalker_dosample),
            subtalker_top_k=pick("subtalker_top_k", subtalker_top_k),
            subtalker_top_p=pick("subtalker_top_p", subtalker_top_p),
            subtalker_temperature=pick("subtalker_temperature", subtalker_temperature),
            max_new_tokens=pick("max_new_tokens", max_new_tokens),
        )
        return merged

    # voice clone model
    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: Union[AudioLike, List[AudioLike]],
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
    ) -> List[VoiceClonePromptItem]:
        """Build voice-clone prompt items from reference audio."""
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with tts_model_type={self.model.tts_model_type} "
                "does not support create_voice_clone_prompt"
            )

        ref_audio_list = self._ensure_list(ref_audio)
        ref_text_list = self._ensure_list(ref_text) if isinstance(ref_text, list) else ([ref_text] * len(ref_audio_list))
        xvec_list = self._ensure_list(x_vector_only_mode) if isinstance(x_vector_only_mode, list) else ([x_vector_only_mode] * len(ref_audio_list))

        if len(ref_text_list) != len(ref_audio_list) or len(xvec_list) != len(ref_audio_list):
            raise ValueError(
                f"Batch size mismatch: ref_audio={len(ref_audio_list)}, ref_text={len(ref_text_list)}, x_vector_only_mode={len(xvec_list)}"
            )

        normalized = self._normalize_audio_inputs(ref_audio_list)

        ref_wavs_for_code: List[np.ndarray] = []
        ref_sr_for_code: List[int] = []
        for wav, sr in normalized:
            ref_wavs_for_code.append(wav)
            ref_sr_for_code.append(sr)

        if len(set(ref_sr_for_code)) == 1:
            enc = self.model.speech_tokenizer.encode(ref_wavs_for_code, sr=ref_sr_for_code[0])
            ref_codes = enc.audio_codes
        else:
            ref_codes = []
            for wav, sr in normalized:
                ref_codes.append(self.model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0])

        items: List[VoiceClonePromptItem] = []
        for i, ((wav, sr), code, rtext, xvec_only) in enumerate(zip(normalized, ref_codes, ref_text_list, xvec_list)):
            if not xvec_only:
                if rtext is None or rtext == "":
                    raise ValueError(f"ref_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}")

            wav_resample = wav
            if sr != self.model.speaker_encoder_sample_rate:
                wav_resample = librosa.resample(y=wav_resample.astype(np.float32),
                                           orig_sr=int(sr),
                                           target_sr=self.model.speaker_encoder_sample_rate)

            spk_emb = self.model.extract_speaker_embedding(audio=wav_resample,
                                                           sr=self.model.speaker_encoder_sample_rate)

            items.append(
                VoiceClonePromptItem(
                    ref_code=None if xvec_only else code,
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=bool(xvec_only),
                    icl_mode=bool(not xvec_only),
                    ref_text=rtext,
                )
            )
        return items

    def _prompt_items_to_voice_clone_prompt(self, items: List[VoiceClonePromptItem]) -> Dict[str, Any]:
        return dict(
            ref_code=[it.ref_code for it in items],
            ref_spk_embedding=[it.ref_spk_embedding for it in items],
            x_vector_only_mode=[it.x_vector_only_mode for it in items],
            icl_mode=[it.icl_mode for it in items],
        )

    # voice clone model
    @torch.no_grad()
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """Voice clone speech using the Base model."""
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with tts_model_type={self.model.tts_model_type} "
                "does not support generate_voice_clone"
            )

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(texts) != len(languages):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}")

        self._validate_languages(languages)

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
            prompt_items = self.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            if isinstance(voice_clone_prompt, list):
                prompt_items = voice_clone_prompt
                if len(prompt_items) == 1 and len(texts) > 1:
                    prompt_items = prompt_items * len(texts)
                if len(prompt_items) != len(texts):
                    raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
                ref_texts_for_ids = [it.ref_text for it in prompt_items]
            else:
                voice_clone_prompt_dict = voice_clone_prompt
                ref_texts_for_ids = None

        input_texts = [self._build_assistant_text(t) for t in texts]
        input_ids = self._tokenize_texts(input_texts)

        ref_ids = None
        if ref_texts_for_ids is not None:
            ref_ids = []
            for i, rt in enumerate(ref_texts_for_ids):
                if rt is None or rt == "":
                    ref_ids.append(None)
                else:
                    ref_tok = self._tokenize_texts([self._build_ref_text(rt)])[0]
                    ref_ids.append(ref_tok)

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        wavs_all, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_for_decode])

        wavs_out: List[np.ndarray] = []
        for i, wav in enumerate(wavs_all):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                ref_len = int(ref_code_list[i].shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wavs_out.append(wav[cut:])
            else:
                wavs_out.append(wav)

        return wavs_out, fs

    @torch.inference_mode()
    def stream_generate_voice_clone(
        self,
        text: str,
        language: str = None,
        ref_audio: Optional[AudioLike] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        voice_clone_prompt: Optional[Union[Dict[str, Any], VoiceClonePromptItem]] = None,
        non_streaming_mode: bool = False,
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 0,
        max_frames: int = 10000,
        use_optimized_decode: bool = True,
        **kwargs,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Stream voice clone speech generation, yielding PCM chunks."""
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with tts_model_type={self.model.tts_model_type} "
                "does not support stream_generate_voice_clone"
            )

        if isinstance(text, list):
            raise ValueError("stream_generate_voice_clone only supports single text, not batch")

        texts = [text]
        languages = [language if language is not None else "Auto"]
        self._validate_languages(languages)

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either voice_clone_prompt or ref_audio must be provided")
            prompt_items = self.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode
            )
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [prompt_items[0].ref_text]
        elif isinstance(voice_clone_prompt, VoiceClonePromptItem):
            prompt_items = [voice_clone_prompt]
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [voice_clone_prompt.ref_text]
        elif isinstance(voice_clone_prompt, list):
            prompt_items = voice_clone_prompt
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [prompt_items[0].ref_text]
        else:
            voice_clone_prompt_dict = voice_clone_prompt
            ref_texts_for_ids = None

        input_texts = [self._build_assistant_text(t) for t in texts]
        input_ids = self._tokenize_texts(input_texts)

        ref_ids = None
        if ref_texts_for_ids is not None:
            ref_ids = []
            for rt in ref_texts_for_ids:
                if rt is None or rt == "":
                    ref_ids.append(None)
                else:
                    ref_tok = self._tokenize_texts([self._build_ref_text(rt)])[0]
                    ref_ids.append(ref_tok)

        gen_kwargs = self._merge_generate_kwargs(**kwargs)
        supported_params = {
            "do_sample", "top_k", "top_p", "temperature",
            "subtalker_dosample", "subtalker_top_k", "subtalker_top_p", "subtalker_temperature"
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in supported_params}

        for chunk, sr in self.model.stream_generate_pcm(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            overlap_samples=overlap_samples,
            max_frames=max_frames,
            use_optimized_decode=use_optimized_decode,
            **gen_kwargs,
        ):
            yield chunk, sr

    # voice design model
    @torch.no_grad()
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        instruct: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """Generate speech with the VoiceDesign model."""
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with tts_model_type={self.model.tts_model_type} "
                "does not support generate_voice_design"
            )

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        instructs = self._ensure_list(instruct)

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(instructs)):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}, instruct={len(instructs)}")

        self._validate_languages(languages)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs

    # custom voice model
    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """Generate speech with the CustomVoice model using a predefined speaker."""
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with tts_model_type={self.model.tts_model_type} "
                "does not support generate_custom_voice"
            )

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        speakers = self._ensure_list(speaker)
        if self.model.tts_model_size in "0b6": # for 0b6 model, instruct is not supported
            instruct = None
        instructs = self._ensure_list(instruct) if isinstance(instruct, list) else ([instruct] * len(texts) if instruct is not None else [""] * len(texts))

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(speakers) == 1 and len(texts) > 1:
            speakers = speakers * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(speakers) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}, speaker={len(speakers)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)
        self._validate_speakers(speakers)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs


    # custom voice model - streaming (PATCH: not in upstream dffdeeq)
    @torch.no_grad()
    def stream_generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str = None,
        instruct: Optional[str] = None,
        non_streaming_mode: bool = False,
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 0,
        max_frames: int = 10000,
        use_optimized_decode: bool = True,
        **kwargs,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Stream custom voice speech generation, yielding PCM chunks as they are generated.

        This method is modeled after stream_generate_voice_clone() but works with
        the CustomVoice model (predefined speakers instead of reference audio).

        NOTE: Not in upstream dffdeeq — this is our addition.
        NOTE: Only supports single-sample generation (no batching).

        Yields:
            Tuple[np.ndarray, int]: (pcm_chunk as float32 array, sample_rate)
        """
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with tts_model_type={self.model.tts_model_type} "
                "does not support stream_generate_custom_voice"
            )

        if isinstance(text, list):
            raise ValueError("stream_generate_custom_voice only supports single text, not batch")

        texts = [text]
        languages = [language if language is not None else "Auto"]
        speakers = [speaker]

        self._validate_languages(languages)
        self._validate_speakers(speakers)

        if self.model.tts_model_size in "0b6":
            instruct = None
        instructs = [instruct if instruct is not None else ""]

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)
        supported_params = {
            "do_sample", "top_k", "top_p", "temperature",
            "subtalker_dosample", "subtalker_top_k", "subtalker_top_p", "subtalker_temperature"
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in supported_params}

        for chunk, sr in self.model.stream_generate_pcm(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            speakers=speakers,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            overlap_samples=overlap_samples,
            max_frames=max_frames,
            use_optimized_decode=use_optimized_decode,
            **gen_kwargs,
        ):
            yield chunk, sr


    def get_supported_speakers(self) -> Optional[List[str]]:
        supported = self._supported_speakers_set()
        if supported is None:
            return None
        return sorted(supported)

    def get_supported_languages(self) -> Optional[List[str]]:
        supported = self._supported_languages_set()
        if supported is None:
            return None
        return sorted(supported)
