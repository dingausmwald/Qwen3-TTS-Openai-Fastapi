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
- **Left padding in `decode_padded()` uses codec silence instead of code 0.**
  Code 0 is a regular codebook entry, not silence. In ICL voice cloning the
  reference codes and the new codes are decoded together; with a long reference
  (e.g. 25.8 s = 322 frames) the new speech starts right after the 300-frame
  chunk boundary, in a chunk that was left-padded with ~230 frames of code 0.
  The decoder (sliding-window attention) saw those fake frames directly before
  the start of the answer, which audibly distorted the first syllables in about
  3 of 10 generations. Padding with real codec silence removes it without extra
  memory. (Raising `chunk_size` to 600 also removed it, but cost ~1.9 GB GTT.)
  - `silence_codes_12hz.pt` (`[38, 16]`, int64) is the 12 Hz tokenizer's encoding
    of 3 s of digital silence at 24 kHz; frames `[5:-5]` are tiled as padding.
  - Looked up next to the module, or via `QWEN3_TTS_SILENCE_CODES`.
  - **No fallback to zeros:** a missing or mismatched file raises at the first
    decode (the warmup), so a broken install fails loudly instead of distorting.
  - Startup log confirms it: `Decode padding uses codec silence from ... (28 frames)`.
- Upstream QwenLM fix `5f8581d`: causal padding in the transposed conv
  (`left_pad = 0`, trim only on the right).
- Upstream QwenLM fix `6cafe55`: code 0 is valid, not padding
  (`audio_lengths` counts `> -1`, codes are clamped to `>= 0` before decode).

### qwen3_tts_tokenizer.py
- Upstream QwenLM fix `6cafe55`: batch padding uses `-1` instead of `0`, so a real
  code 0 is no longer mistaken for padding.

### configuration_qwen3_tts.py
- "... is None. Initializing ... with default values" logged only once instead of
  per request.

### configuration_qwen3_tts_tokenizer_v2.py
- `code_predictor_config` logged only once instead of per-request.

## Applying manually

If you installed dffdeeq via pip instead of using the Dockerfile:

```bash
QWEN_TTS=$(python -c "import qwen_tts; print(qwen_tts.__path__[0])")
cp patches/dffdeeq/qwen3_tts_model.py "$QWEN_TTS/inference/"
cp patches/dffdeeq/modeling_qwen3_tts_tokenizer_v2.py "$QWEN_TTS/core/tokenizer_12hz/"
cp patches/dffdeeq/configuration_qwen3_tts_tokenizer_v2.py "$QWEN_TTS/core/tokenizer_12hz/"
cp patches/dffdeeq/silence_codes_12hz.pt "$QWEN_TTS/core/tokenizer_12hz/"
cp patches/dffdeeq/qwen3_tts_tokenizer.py "$QWEN_TTS/inference/"
cp patches/dffdeeq/configuration_qwen3_tts.py "$QWEN_TTS/core/models/"
find "$QWEN_TTS" -name "*.pyc" -delete
```
