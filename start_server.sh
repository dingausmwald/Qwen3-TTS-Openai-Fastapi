#!/bin/bash
# Qwen3-TTS Server Startup Script
#
# Auto-detects GPU vendor (AMD/NVIDIA) and sets appropriate environment variables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect GPU vendor
if command -v rocminfo &>/dev/null || [ -d /opt/rocm ]; then
    echo "Detected AMD ROCm GPU"
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    export MIOPEN_FIND_MODE=FAST
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    export GPU_MAX_ALLOC_PERCENT=100
    export GPU_MAX_HEAP_SIZE=100
    # Fix ROCm bug: multiple HIP streams keep GPU at max clock idle
    # See https://github.com/ROCm/ROCm/issues/2625
    export GPU_MAX_HW_QUEUES=1
elif command -v nvidia-smi &>/dev/null; then
    echo "Detected NVIDIA CUDA GPU"
    # NVIDIA typically needs no special env vars
fi

# Server config
export TTS_BACKEND=optimized
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8880}
export WORKERS=1

# Add streaming library to Python path if installed separately
if [ -d "/opt/qwen3-tts-streaming" ]; then
    export PYTHONPATH="/opt/qwen3-tts-streaming:$SCRIPT_DIR:$PYTHONPATH"
else
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
fi

# Activate venv if present
for venv in /opt/venv .venv venv; do
    if [ -f "$venv/bin/activate" ]; then
        source "$venv/bin/activate" 2>/dev/null
        break
    fi
done

echo "Starting Qwen3-TTS (optimized) on :$PORT"
exec python -m api.main
