# Pipecat Voice Agent

A real-time voice agent using WebRTC, combining local STT (faster-whisper),
any OpenAI-compatible LLM, and Qwen3-TTS for speech synthesis.

## Prerequisites

- Qwen3-TTS server running (port 8880)
- An LLM server with OpenAI-compatible API (e.g. llamaswap, ollama, vllm)
- HTTPS reverse proxy (WebRTC requires secure context)

## Quick Start

```bash
# Copy files to server
scp examples/pipecat/* user@server:~/pipecat/

# Create container
docker run -d --name pipecat \
  --network host \
  -v ~/pipecat:/data:z \
  python:3.13-slim \
  bash -c "apt-get update && apt-get install -y libsndfile1 && \
    python -m venv /data/venv && \
    /data/venv/bin/pip install pipecat-ai[silero,whisper,openai,smallwebrtc] pyyaml && \
    /data/venv/bin/python -m pipecat.runner.run --host 0.0.0.0 --port 7860 /data/app.py"

# Pre-download Whisper model (~500 MB)
docker exec pipecat /data/venv/bin/python -c "
from faster_whisper import WhisperModel
WhisperModel('small', device='cpu', compute_type='default')
print('Done')
"
docker restart pipecat
```

## Configuration

Edit `config.yaml` and restart the container.

Key settings:
- `tts.voice`: Use `alloy`, `echo`, etc. for built-in voices, or `clone:ProfileName` for voice cloning
- `tts.streaming`: `true` for real-time audio streaming, `false` for buffered playback
- `tts.model`: Add language suffix for language hints: `tts-1-de`, `tts-1-ja`, etc.
- `stt.model`: Whisper model size (tiny/small/medium/large-v3)

## HTTPS Setup

WebRTC requires a secure context. Use a reverse proxy like Caddy:

```
# /etc/caddy/Caddyfile
:8446 {
    tls /path/to/cert.pem /path/to/key.pem
    reverse_proxy localhost:7860
}
```

Access at `https://your-server:8446`.
