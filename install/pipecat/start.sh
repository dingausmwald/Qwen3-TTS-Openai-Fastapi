#!/bin/bash
set -e

PIPECAT_DIR=/data

# Install system dependencies (OpenCV, audio libs) — only once
DEPS_MARKER="$PIPECAT_DIR/venv/.system_deps_installed"
if [ ! -f "$DEPS_MARKER" ]; then
    echo "Installing system dependencies..."
    apt-get update -qq
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 > /dev/null
    rm -rf /var/lib/apt/lists/*
    touch "$DEPS_MARKER"
fi

# Create venv if not exists
if [ ! -d "$PIPECAT_DIR/venv" ]; then
    echo "Creating Python venv..."
    python3 -m venv "$PIPECAT_DIR/venv"
fi

source "$PIPECAT_DIR/venv/bin/activate"

# Install/update if requirements changed
if [ -f "$PIPECAT_DIR/requirements.txt" ]; then
    HASH_FILE="$PIPECAT_DIR/venv/.requirements_hash"
    CURRENT_HASH=$(md5sum "$PIPECAT_DIR/requirements.txt" | cut -d' ' -f1)
    if [ ! -f "$HASH_FILE" ] || [ "$(cat $HASH_FILE)" != "$CURRENT_HASH" ]; then
        echo "Installing/updating requirements..."
        pip install -r "$PIPECAT_DIR/requirements.txt"
        echo "$CURRENT_HASH" > "$HASH_FILE"
    fi
fi

# Start pipecat app if it exists
if [ -f "$PIPECAT_DIR/app.py" ]; then
    echo "Starting Pipecat..."
    exec python3 "$PIPECAT_DIR/app.py" -t webrtc --host 0.0.0.0 --port 7870
else
    echo "No app.py found, waiting..."
    exec sleep infinity
fi
