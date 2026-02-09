# Qwen3-TTS OpenAI-compatible Server — NVIDIA CUDA
#
# Build:  docker build -t qwen3-tts .
# Run:    docker run --gpus all -p 8880:8880 -v ~/qwen3-tts:/root/qwen3-tts qwen3-tts

FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /opt/qwen3-tts

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install flash attention (NVIDIA CUDA)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Copy application code (includes qwen_tts/ with streaming patches applied)
COPY . .

# Install the package (makes qwen_tts importable)
RUN pip install --no-cache-dir -e .

# Default config location
RUN mkdir -p /root/qwen3-tts/voice_library
COPY config.yaml /root/qwen3-tts/config.yaml

ENV TTS_BACKEND=optimized \
    HOST=0.0.0.0 \
    PORT=8880 \
    WORKERS=1

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

CMD ["python", "-m", "api.main"]
