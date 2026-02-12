"""
Standalone Audio Cleanup Nodes for ComfyUI
- DeepFilterNet3: AI speech enhancement (best for TTS artifacts)
- UVR DeReverb: AI reverb/echo removal
"""

import os
import sys
import types
import tempfile
import numpy as np
import torch
import soundfile as sf

# --- Patch torchaudio for DeepFilterNet compatibility ---
import torchaudio
if not hasattr(torchaudio, "backend"):
    _backend_mod = types.ModuleType("torchaudio.backend")
    _common_mod = types.ModuleType("torchaudio.backend.common")
    class _AudioMetaData:
        def __init__(self, sample_rate=0, num_frames=0, num_channels=0, bits_per_sample=0, encoding=None):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding
    _common_mod.AudioMetaData = _AudioMetaData
    _backend_mod.common = _common_mod
    torchaudio.backend = _backend_mod
    sys.modules["torchaudio.backend"] = _backend_mod
    sys.modules["torchaudio.backend.common"] = _common_mod


# ── DeepFilterNet Node ──────────────────────────────────────────────

_df_model = None
_df_state = None

def _get_df():
    global _df_model, _df_state
    if _df_model is None:
        from df.enhance import init_df
        _df_model, _df_state, _ = init_df()
    return _df_model, _df_state


class DeepFilterNetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "attenuation_limit_db": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Max noise attenuation in dB. 0 = no limit (full enhancement)."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("enhanced_audio",)
    FUNCTION = "process"
    CATEGORY = "audio"

    def process(self, audio, attenuation_limit_db=0.0):
        from df.enhance import enhance
        import librosa

        model, df_state = _get_df()
        target_sr = df_state.sr()  # 48000

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if isinstance(waveform, torch.Tensor):
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            wav_np = waveform.cpu().numpy().squeeze()
        else:
            wav_np = np.asarray(waveform).squeeze()

        # Resample to 48kHz if needed
        if sample_rate != target_sr:
            wav_np = librosa.resample(wav_np, orig_sr=sample_rate, target_sr=target_sr)

        # DeepFilterNet expects [channels, samples] torch tensor
        audio_tensor = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)

        # Enhance
        atten_lim = attenuation_limit_db if attenuation_limit_db > 0 else None
        enhanced = enhance(model, df_state, audio_tensor, atten_lim_db=atten_lim)

        enhanced_np = enhanced.squeeze().cpu().numpy()

        # Resample back to original sample rate
        if sample_rate != target_sr:
            enhanced_np = librosa.resample(enhanced_np, orig_sr=target_sr, target_sr=sample_rate)

        out = torch.tensor(enhanced_np, dtype=torch.float32)
        if out.dim() == 1:
            out = out.unsqueeze(0).unsqueeze(0)

        return ({"waveform": out, "sample_rate": sample_rate},)


# ── UVR DeReverb Node ───────────────────────────────────────────────

_separator_instance = None
_loaded_model = None

def _get_separator(model_name):
    global _separator_instance, _loaded_model
    from audio_separator.separator import Separator
    if _separator_instance is None or _loaded_model != model_name:
        _separator_instance = Separator(output_format="WAV")
        _separator_instance.load_model(model_name)
        _loaded_model = model_name
    return _separator_instance


class AudioDereverbNode:
    MODELS = [
        "UVR-DeEcho-DeReverb.pth",
        "UVR-De-Echo-Normal.pth",
        "UVR-De-Echo-Aggressive.pth",
        "UVR-DeNoise.pth",
        "UVR-DeNoise-Lite.pth",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (cls.MODELS, {"default": "UVR-DeEcho-DeReverb.pth"}),
            },
            "optional": {
                "aggressiveness": ("INT", {"default": 10, "min": 0, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("cleaned", "removed")
    FUNCTION = "process"
    CATEGORY = "audio"

    def process(self, audio, model="UVR-DeEcho-DeReverb.pth", aggressiveness=10):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if isinstance(waveform, torch.Tensor):
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            wav_np = waveform.cpu().numpy()
        else:
            wav_np = waveform

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "input.wav")
            if wav_np.ndim == 1:
                sf.write(in_path, wav_np, sample_rate)
            else:
                sf.write(in_path, wav_np.T, sample_rate)

            sep = _get_separator(model)
            sep.output_dir = tmpdir
            sep.aggressiveness = aggressiveness
            output_files = sep.separate(in_path)

            results = {}
            for f in output_files:
                data, sr = sf.read(f)
                if data.ndim > 1:
                    data = data[:, 0]
                basename = os.path.basename(f).lower()
                if "no reverb" in basename or "no echo" in basename or "no noise" in basename:
                    results["cleaned"] = (data, sr)
                else:
                    results["removed"] = (data, sr)

            if "cleaned" not in results:
                all_stems = []
                for f in output_files:
                    data, sr = sf.read(f)
                    if data.ndim > 1:
                        data = data[:, 0]
                    all_stems.append((data, sr))
                results["cleaned"] = all_stems[0]
                results["removed"] = all_stems[1] if len(all_stems) > 1 else all_stems[0]

        def to_audio_dict(data, sr):
            t = torch.tensor(data, dtype=torch.float32)
            if t.dim() == 1:
                t = t.unsqueeze(0).unsqueeze(0)
            return {"waveform": t, "sample_rate": sr}

        return (
            to_audio_dict(*results["cleaned"]),
            to_audio_dict(*results["removed"]),
        )


NODE_CLASS_MAPPINGS = {
    "DeepFilterNetNode": DeepFilterNetNode,
    "AudioDereverbNode": AudioDereverbNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepFilterNetNode": "DeepFilterNet3 (Speech Enhancement)",
    "AudioDereverbNode": "Audio Dereverb/DeEcho (UVR)",
}
