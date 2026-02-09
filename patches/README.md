# dffdeeq patches

These patches are already applied in this fork's `qwen_tts/` directory.
This directory is a reference for users who installed
[dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming)
separately via pip and want to apply the fixes manually.

## Changes

### qwen3_tts_model.py
- Added `stream_generate_custom_voice()` method for real-time streaming with the
  CustomVoice model. dffdeeq only provides `stream_generate_voice_clone()` for the
  Base model.

### modeling_qwen3_tts_tokenizer_v2.py
- `chunked_decode()` uses `decode_padded()` with fixed tensor sizes to prevent
  torch.compile recompilation on every call.
- `+ 0` trick on decode output to force new tensor allocation. Without this, CUDA
  graphs overwrite the previous chunk's output tensor. `.clone()` is insufficient
  because PyTorch optimizes it away.
- `torch.compiler.cudagraph_mark_step_begin()` before each decode to mark previous
  outputs as complete.

### configuration_qwen3_tts_tokenizer_v2.py
- `code_predictor_config` logged only once instead of per-request.

## Applying manually

If you installed dffdeeq via pip instead of using the Dockerfile:

```bash
QWEN_TTS=$(python -c "import qwen_tts; print(qwen_tts.__path__[0])")
cp patches/dffdeeq/qwen3_tts_model.py "$QWEN_TTS/inference/"
cp patches/dffdeeq/modeling_qwen3_tts_tokenizer_v2.py "$QWEN_TTS/core/tokenizer_12hz/"
cp patches/dffdeeq/configuration_qwen3_tts_tokenizer_v2.py "$QWEN_TTS/core/tokenizer_12hz/"
find "$QWEN_TTS" -name "*.pyc" -delete
```
