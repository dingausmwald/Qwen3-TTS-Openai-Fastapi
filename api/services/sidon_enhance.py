"""Sidon speech restoration for reference recordings.

Runs in-process on CPU: the traced models need nothing that this container does
not already ship (torchaudio, transformers, huggingface_hub). Roughly 3x realtime,
so a 25 s reference takes about 8 s. Loaded lazily on first use.

The preprocessing chain is replicated verbatim from the official demo Space
(https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta). Do not "tidy" it:
the normalisation, the 50 Hz highpass and the padding constants are part of the
contract with the traced model, and getting one wrong degrades quality silently.
"""
import io
import logging
import threading

import numpy as np
import soundfile as sf
import torch
import torchaudio
import transformers
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

REPO = "sarulab-speech/sidon-v0.1"
OUTPUT_SR = 48_000

_lock = threading.Lock()
_state = {}


def _ensure_loaded():
    """Load models once. Held for the process lifetime (~1 GB, CPU)."""
    if _state:
        return _state
    with _lock:
        if _state:
            return _state
        import time
        t0 = time.time()
        fe_path = hf_hub_download(REPO, filename="feature_extractor_cpu.pt")
        dec_path = hf_hub_download(REPO, filename="decoder_cpu.pt")
        _state["fe"] = torch.jit.load(fe_path, map_location="cpu")
        _state["decoder"] = torch.jit.load(dec_path, map_location="cpu")
        _state["pre"] = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        logger.info("Sidon loaded in %.1fs", time.time() - t0)
    return _state


@torch.inference_mode()
def restore(audio_bytes: bytes) -> bytes:
    """Restore one recording. Returns 48 kHz mono WAV bytes."""
    s = _ensure_loaded()
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))

    waveform = 0.9 * (waveform / np.abs(waveform).max())
    target_n_samples = int(OUTPUT_SR / sample_rate * waveform.shape[0])

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=1)

    waveform = waveform.view(1, -1)
    wav = torchaudio.functional.highpass_biquad(waveform, sample_rate, 50)
    wav_16k = torchaudio.functional.resample(wav, sample_rate, 16_000)

    restoreds = []
    feature_cache = None
    wav_16k = torch.nn.functional.pad(wav_16k, (0, 24000))
    for chunk in wav_16k.view(-1).split(16000 * 96):
        inputs = s["pre"](torch.nn.functional.pad(chunk, (160, 160)), return_tensors="pt")
        feature = s["fe"](inputs["input_features"])["last_hidden_state"]
        if feature_cache is not None:
            feature = torch.cat([feature_cache, feature], dim=1)
        restoreds.append(s["decoder"](feature.transpose(1, 2)).view(-1)[:-960])
        feature_cache = feature[:, -1:]

    restored = torch.cat(restoreds, dim=0)
    out = (restored.view(-1, 1).cpu().numpy() * 32767).astype(np.int16)[:target_n_samples]

    buf = io.BytesIO()
    sf.write(buf, out, OUTPUT_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()
