# Qwen3-TTS OpenAI-compatible Server

An OpenAI-compatible TTS server for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
with real-time audio streaming, voice cloning, and torch.compile optimizations.

Based on [groxaxo/Qwen3-TTS-Openai-Fastapi](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi)
and [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming).


## What this fork adds

- **Real-time audio streaming** — `stream: true` in the request body yields PCM
  chunks as the model generates them, instead of waiting for the full audio.
  Works for both built-in voices and voice cloning.
- **Voice library** — Save voice profiles and use them with `voice: "clone:MyVoice"`.
  The server auto-switches between CustomVoice and Base models as needed.
- **Voice prompt caching** — Speaker embeddings are computed once per profile and
  reused across requests. Saves ~0.7s per voice clone request.
- **torch.compile + CUDA graphs** — Configurable via `config.yaml`. The `optimized`
  backend compiles both streaming and non-streaming code paths on startup.
- **Pipecat integration** — Example voice agent with WebRTC, local Whisper STT,
  and streaming TTS (see [examples/pipecat](examples/pipecat/)).


## Compatibility

| Platform | Status |
|----------|--------|
| AMD ROCm (tested: Strix Halo gfx1151) | Tested, optimized |
| NVIDIA CUDA (Ampere, Ada, Blackwell) | Should work, not yet verified |
| CPU | Use groxaxo's `pytorch_cpu` backend instead |

The core features (streaming API, voice library, caching) are platform-agnostic.
torch.compile and CUDA graphs work on both CUDA and ROCm.

NVIDIA users: if you encounter issues, please open an issue. The streaming library
(dffdeeq) was originally developed and benchmarked on RTX 5090.


## Quick Start

```bash
# NVIDIA
docker compose up -d

# AMD ROCm
docker compose -f docker-compose.rocm.yml up -d
```

Models are downloaded from HuggingFace automatically on first start (~2.4 GB each).
First startup takes ~75 seconds for torch.compile warmup.

The server listens on port **8880** by default.


## Usage

### OpenAI-compatible API

```bash
# Non-streaming (compatible with OpenWebUI, SillyTavern, etc.)
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "alloy", "model": "tts-1"}' \
  --output speech.mp3

# Real-time streaming (for low-latency applications)
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "alloy", "model": "tts-1", "stream": true, "response_format": "pcm"}' \
  --output speech.pcm

# Voice cloning (auto-switches to Base model)
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "clone:MyVoice", "model": "tts-1-en"}' \
  --output speech.mp3
```

### Language hints

Append a language code to the model name for language-specific synthesis:

| Model | Language |
|-------|----------|
| `tts-1-en` | English |
| `tts-1-de` | German |
| `tts-1-zh` | Chinese |
| `tts-1-ja` | Japanese |
| `tts-1-ko` | Korean |
| `tts-1-fr` | French |
| `tts-1-es` | Spanish |
| `tts-1-ru` | Russian |
| `tts-1-pt` | Portuguese |
| `tts-1-it` | Italian |

### Voice mapping

| OpenAI voice | Qwen3-TTS speaker |
|--------------|-------------------|
| alloy | Vivian |
| echo | Ryan |
| fable | Serena |
| nova | Aiden |
| onyx | Eric |
| shimmer | Dylan |


## Configuration

The server reads `~/qwen3-tts/config.yaml` (or `$TTS_CONFIG`).

Key settings:

```yaml
default_model: 0.6B-CustomVoice

optimization:
  compile_mode: max-autotune     # torch.compile mode
  use_cuda_graphs: true          # CUDA graph capture
  streaming:
    decode_window_frames: 72     # AMD: use 72 (not 64 or 80). NVIDIA: 64 or 80 also work.
    emit_every_frames: 24        # ~3x redundancy ratio is optimal
```

See [config.yaml](config.yaml) for all options with comments.


## Performance

Benchmarked on AMD Radeon 8060S (Strix Halo), 0.6B model, after warmup:

| Mode | RTF | TTFB | Notes |
|------|-----|------|-------|
| Non-streaming, CustomVoice | 0.53-0.54x | full generation time | Best for OpenWebUI |
| Non-streaming, Voice Clone | 0.55-0.56x | full generation time | With cached prompt |
| Streaming, CustomVoice | 0.57x | ~0.3s | 6-8 chunks per request |
| Streaming, Voice Clone | 0.57x | ~0.3s | With cached prompt |

RTF = Real-Time Factor. Lower is faster. 0.54x means audio is generated ~1.85x
faster than real-time.

### Streaming tuning

The `decode_window_frames / emit_every_frames` ratio controls the trade-off between
latency (TTFB) and throughput (RTF):

- **~3x redundancy** is optimal (e.g. 72/24, 150/50, 300/100)
- Smaller windows = lower TTFB at the same ratio
- RTF is nearly identical regardless of absolute window size at the same ratio

AMD ROCm note: `decode_window_frames` values of 64 or less, and exactly 80, trigger
a CUDA graph capture bug that causes 5-10x slowdown. Use 72 or 84+.


## Architecture

```
Client (OpenWebUI, Pipecat, curl, ...)
    |
    v
FastAPI Router (openai_compatible.py)
    |  - OpenAI API compatibility
    |  - Voice library (clone: prefix)
    |  - Ref audio caching
    |  - Stream/non-stream routing
    v
Optimized Backend (optimized_backend.py)
    |  - Model switching (CustomVoice <-> Base)
    |  - torch.compile + CUDA graphs
    |  - Voice prompt caching
    v
dffdeeq/Qwen3-TTS-streaming
    |  - stream_generate_pcm()
    |  - stream_generate_custom_voice() (our patch)
    |  - create_voice_clone_prompt()
    v
Qwen3-TTS Model (HuggingFace)
```


## Project structure

```
api/
  backends/
    optimized_backend.py    # Optimized backend: compile, cache, stream, model switch
    factory.py              # Backend selection (optimized / official / vllm / pytorch / openvino)
  routers/
    openai_compatible.py    # OpenAI API endpoints with streaming + voice library
  structures/
    schemas.py              # Request/response schemas
  services/                 # From groxaxo: text processing, audio encoding
qwen_tts/                   # Inference library (dffdeeq fork with patches applied)
  inference/
    qwen3_tts_model.py      # + stream_generate_custom_voice()
  core/tokenizer_12hz/
    modeling_qwen3_tts_tokenizer_v2.py  # + CUDA graph tensor fixes
    configuration_qwen3_tts_tokenizer_v2.py
patches/
  dffdeeq/                  # Patch reference (for upstream dffdeeq users)
examples/
  pipecat/                  # WebRTC voice agent example
config.yaml                 # Server configuration
start_server.sh             # Auto-detects AMD/NVIDIA, sets env vars
Dockerfile                  # NVIDIA CUDA
Dockerfile.rocm             # AMD ROCm
docker-compose.yml          # NVIDIA
docker-compose.rocm.yml     # AMD
```

Files in `api/services/`, `api/main.py`, and groxaxo's other backends come from
[groxaxo](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi) upstream.
The `qwen_tts/` library is from [dffdeeq](https://github.com/dffdeeq/Qwen3-TTS-streaming)
with patches for CustomVoice streaming and CUDA graph fixes applied directly.


## Manual installation (without Docker)

```bash
# 1. Clone this repo
git clone https://github.com/dingausmwald/Qwen3-TTS-Openai-Fastapi.git
cd Qwen3-TTS-Openai-Fastapi

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Install flash attention
# NVIDIA:
pip install flash-attn --no-build-isolation
# AMD:
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install flash-attn --no-build-isolation

# 4. Copy config
mkdir -p ~/qwen3-tts/voice_library
cp config.yaml ~/qwen3-tts/config.yaml

# 5. Start
./start_server.sh
```


## Voice Library

To use voice cloning with saved profiles, create a profile directory:

```
~/qwen3-tts/voice_library/profiles/
  my_voice/
    meta.json
    reference.wav
```

`meta.json` format:
```json
{
  "name": "MyVoice",
  "profile_id": "my_voice",
  "ref_audio_filename": "reference.wav",
  "ref_text": "Optional transcript of the reference audio.",
  "x_vector_only_mode": false,
  "language": "English",
  "task_type": "Base"
}
```

Then use `voice: "clone:MyVoice"` in API requests. The server auto-switches to
the Base model for voice cloning.


## Acknowledgments

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — The model
- [groxaxo/Qwen3-TTS-Openai-Fastapi](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi) — API server framework
- [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) — Streaming inference library


## License

Apache-2.0, same as the upstream projects.
