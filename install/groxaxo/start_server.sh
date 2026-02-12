#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ROCm / AMD optimizations
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export MIOPEN_FIND_MODE=FAST
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME=$HOME/qwen3-tts/tunableop_results.csv
export GPU_MAX_ALLOC_PERCENT=100
export GPU_MAX_HEAP_SIZE=100
# ROCm firmware bug: HIP streams cause false 100% GPU busy reporting
# See https://github.com/ROCm/ROCm/issues/5107
export GPU_MAX_HW_QUEUES=1

# Server config
export TTS_BACKEND=optimized
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8880}
export WORKERS=1
export PYTHONPATH="/opt/qwen3-tts-streaming:$SCRIPT_DIR:$PYTHONPATH"
export VOICE_LIBRARY_DIR="$HOME/qwen3-tts/voice_library"
export ENABLE_VOICE_STUDIO=true
# Prevent AMD GPU idle downclocking (DPM auto drops to 600MHz after ~20s)
export GPU_KEEPALIVE_INTERVAL=15

# Activate venv (ROCm PyTorch)
source /opt/venv/bin/activate 2>/dev/null

echo "Starting Qwen3-TTS (optimized) on :$PORT"
exec python -m api.main
