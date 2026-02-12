# Qwen3-TTS Installation

OpenAI-kompatible TTS-API mit Qwen3-TTS auf AMD ROCm.

## Voraussetzungen

- Linux mit Podman + Distrobox
- AMD GPU mit ROCm-Support
- Container-Image: `docker.io/kyuz0/amd-strix-halo-comfyui:latest` (enthaelt ROCm + PyTorch in `/opt/venv/`)

## Tech Stack

**Zwei Repos, per Symlink verbunden:**

| Repo | Zweck |
|------|-------|
| [groxaxo/Qwen3-TTS-Openai-Fastapi](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi) | API-Server: FastAPI, OpenAI-kompatible Routers, Voice Studio Web-UI |
| [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) | Modell-Code: `enable_streaming_optimizations()`, `stream_generate_pcm()`, torch.compile, CUDA Graphs, fast codebook |

Groxaxo hat dffdeeqs Optimierungen nicht gemerged. Darum wird groxaxos `qwen_tts/` per Symlink auf dffdeeqs Fork umgelenkt.

**Flash Attention (Dao-AILab):** Muss fuer AMD aus Source gebaut werden (nutzt intern AMDs Triton-Backend via `FLASH_ATTENTION_TRITON_AMD_ENABLE`). Ohne flash_attn faellt PyTorch auf SDPA zurueck — das verursacht RTF-Spikes (Verzoegerungen) in ca. jeder 3.-5. Anfrage.

## Optimierungen & Fixes

### config.yaml (optimization)

| Setting | Wert | Erklaerung |
|---------|------|------------|
| `use_compile` | `true` | Aktiviert `torch.compile()` auf dem Modell. Erster Start ~75s Warmup (Kompilierung), danach schnelle Inferenz (~0.5x RTF). |
| `compile_mode` | `max-autotune` | Aggressivste torch.compile-Stufe. Testet viele Kernel-Varianten beim Warmup. Auf AMD schneller als `reduce-overhead` oder `max-autotune-no-cudagraphs`. |
| `use_cuda_graphs` | `true` | Zeichnet den GPU-Aufrufgraphen auf und replayed ihn, spart CPU-Overhead. Funktioniert fuer **beide** Modelltypen (CustomVoice UND Base/Voice Cloning) mit identischen Settings. NICHT deaktivieren fuer Base — das degradiert RTF von 0.55x auf ~1.0x. |
| `use_fast_codebook` | `true` | Schnellere Codebook-Prediction im Tokenizer. |
| `attention` | `flash_attention_2` | Flash Attention statt SDPA. Verhindert RTF-Spikes (s.o.). |
| `compile_codebook_predictor` | `true` | Kompiliert auch den Codebook-Predictor mit torch.compile. |

### config.yaml (streaming)

| Setting | Wert | Erklaerung |
|---------|------|------------|
| `streaming.decode_window_frames` | `72` | Decoder-Fenster fuer Streaming (Frames). Wird bei jedem Emit-Zyklus komplett decoded. Groesseres Fenster = mehr Kontext, aber hoehere Latenz. **ACHTUNG**: Werte ≤64 und genau 80 sind auf AMD defekt (CUDA Graph Bug), ≥72 (ausser 80) funktionieren. |
| `streaming.emit_every_frames` | `24` | Alle N Frames wird ein PCM-Chunk emitted. Bestimmt TTFB und Chunk-Groesse. 24 Frames × 80ms = ~1.92s Audio pro Chunk. |

**Redundancy Ratio** = `decode_window_frames / emit_every_frames`. Bestimmt wie viel redundante Berechnung pro Emit-Zyklus stattfindet. **~3x ist optimal.** Die Ratio hat mehr Einfluss als die absoluten Werte.

### Code-Fixes in `dffdeeq/`

| Datei | Fix | Problem |
|-------|-----|---------|
| `modeling_qwen3_tts_tokenizer_v2.py` | 1) `chunked_decode()` nutzt `decode_padded()` mit fixer Groesse (chunk_size + left_context_size) | Ohne Fix: Variable Tensor-Shapes bei jedem Aufruf → `torch.compile(dynamic=False)` rekompiliert staendig → RTF-Spikes in ~10% der Requests |
| | 2) `torch.compiler.cudagraph_mark_step_begin()` vor jedem Chunk + `+ 0` am Ergebnis | Ohne Fix: CUDA Graphs ueberschreiben den Output-Tensor des vorherigen Chunks. `.clone()` reicht **nicht** — PyTorch optimiert es weg. `+ 0` erzwingt eine neue Allokation ueber den Arithmetik-Pfad. `cudagraph_mark_step_begin()` markiert vorherige Outputs als abgeschlossen. |
| `configuration_qwen3_tts_tokenizer_v2.py` | `code_predictor_config` wird nur einmal geloggt | Ohne Fix: Konfiguration wird bei jedem Request geloggt → Log-Spam |
| `qwen3_tts_model.py` | `stream_generate_custom_voice()` hinzugefuegt (unser Patch!) | dffdeeq hat nativ nur `stream_generate_voice_clone()` (fuer Voice Cloning mit Base-Modell). Das CustomVoice-Modell braucht eine eigene Streaming-Methode. Modelliert nach `stream_generate_voice_clone()`. |

### Code in `groxaxo/`

| Datei | Funktion |
|-------|----------|
| `optimized_backend.py` | Backend mit torch.compile, Model-Switching (CustomVoice ↔ Base), Streaming. Nutzt dffdeeqs `enable_streaming_optimizations()`. **Voice-Prompt-Cache** fuer Voice Cloning: `create_voice_clone_prompt()` wird einmalig berechnet (~0.76s) und danach aus dem Cache bedient. **Model-Type-Warmup**: Base-Modell nutzt `generate_voice_clone()` mit Dummy-Audio fuer Warmup (statt `generate_custom_voice()`, das auf Base crashed). |
| `factory.py` | Backend-Factory, unterstuetzt `TTS_BACKEND=optimized`. |
| `openai_compatible.py` | Router: `/v1/audio/speech` (mit Streaming), `/v1/audio/voices`, Voice Cloning, `clone:`-Prefix fuer gespeicherte Voice-Profile. **Ref-Audio-Cache**: `sf.read()` wird pro Voice-Profil nur einmalig ausgefuehrt, danach gecacht. Gibt `cache_key=profile_name` an das Backend weiter fuer Voice-Prompt-Caching. |
| `schemas.py` | Pydantic-Schemas mit Voice-Clone-Feldern (`ref_audio`, `ref_text`, `x_vector_only_mode`). |
| `start_server.sh` | Startscript, siehe naechster Abschnitt. |

### start_server.sh — Environment-Variablen

| Variable | Wert | Herkunft | Erklaerung |
|----------|------|----------|------------|
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | flash_attn (Dao-AILab) | Aktiviert AMDs Triton-Backend fuer Flash Attention statt des CUDA-Pfads. |
| `MIOPEN_FIND_MODE` | `FAST` | ROCm/MIOpen | MIOpen (AMDs cuDNN-Equivalent) sucht schneller nach optimalen Kernels. Ohne: langsamer Benchmark beim ersten Lauf. |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | PyTorch ROCm | Aktiviert experimentelle AOTriton-Kernels (Ahead-of-Time kompilierte Triton-Ops). Schnellere Attention-Berechnung. |
| `GPU_MAX_ALLOC_PERCENT` | `100` | ROCm Runtime | Erlaubt der GPU bis zu 100% des Speichers zu allokieren. Default ist konservativer. |
| `GPU_MAX_HEAP_SIZE` | `100` | ROCm Runtime | Erlaubt den GPU-Heap bis zu 100% des Speichers zu nutzen. |
| `GPU_MAX_HW_QUEUES` | `1` | ROCm Runtime | Begrenzt auf eine Hardware-Queue. Ohne diesen Fix halten CUDA Graphs die GPU ueber mehrere HIP-Streams auf 100% Takt / ~25W idle. Mit Fix: ~6W idle. Siehe [ROCm#2625](https://github.com/ROCm/ROCm/issues/2625). |
| `TTS_BACKEND` | `optimized` | Eigener Code | Waehlt das optimierte Backend (`optimized_backend.py`) statt des groxaxo-Defaults. |
| `PYTHONPATH` | `/opt/qwen3-tts-streaming:...` | Eigener Code | Stellt sicher, dass dffdeeqs Modell-Code importierbar ist. |

### Voice-Prompt-Caching (Latenz-Optimierung)

Bei Voice Cloning (Stimmprofil via `clone:Name`) muss fuer jede Anfrage ein Speaker-Embedding berechnet werden (`ref_code` + `ref_spk_embedding` / x-vector). Das dauert ~0.76s pro Anfrage. dffdeeq bietet dafuer nativ:

1. `create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode)` → gibt `VoiceClonePromptItem`-Liste zurueck (in-memory Tensoren, keine Datei)
2. `generate_voice_clone(text, voice_clone_prompt=...)` → nutzt das vorberechnete Embedding

**Zwei Cache-Ebenen implementiert:**

| Ebene | Wo | Was wird gecacht | Ersparnis |
|-------|-----|-----------------|-----------|
| `_ref_audio_cache` | `openai_compatible.py` | `sf.read()` Ergebnis pro Voice-Profil | Disk-I/O (~10ms) |
| `_voice_prompt_cache` | `optimized_backend.py` | `VoiceClonePromptItem` (Speaker-Embedding Tensoren) | GPU-Berechnung (~0.76s) |

Cache wird automatisch geleert bei Model-Switch (Base → CustomVoice oder umgekehrt).

**Effekt bei OpenWebUI (5 Saetze seriell):** 9.96s → 8.90s (~10% schneller). Nur der erste Request pro Stimmprofil berechnet das Embedding, alle folgenden nutzen den Cache.

### Model-Type-Warmup (500-Error-Fix)

Beim Model-Switch von CustomVoice auf Base (Voice Cloning) muss der Warmup zum Modelltyp passen:
- **CustomVoice**: `generate_custom_voice(text, speaker, language)` — nutzt eingebaute Speaker
- **Base**: `generate_voice_clone(text, ref_audio, language, x_vector_only_mode=True)` — braucht Referenz-Audio

Ohne diesen Fix: Erster Voice-Clone-Request nach Model-Switch gibt immer 500 Internal Server Error, weil der Warmup `generate_custom_voice()` aufruft, was auf dem Base-Modell fehlschlaegt.

**Fix in `optimized_backend.py`**: `_ensure_model_loaded()` prueft `model_info.type` und nutzt fuer Base-Modelle `generate_voice_clone()` mit einem Dummy-Sinus-Audio (440Hz, 1s) fuer den Warmup.

### Performance-Kennzahlen (nach allen Fixes)

| Metrik | CustomVoice | Voice Cloning (Base) |
|--------|-------------|---------------------|
| Non-Streaming RTF | 0.53-0.54x | 0.55-0.56x |
| **Streaming RTF** | **0.57x** | **0.57x** |
| **Streaming TTFB** | **~1.06s** | **~1.1s** |
| Spikes | 0% (nach Flash-Attn + Spike-Fixes) | 0% |
| Warmup (erster Start, torch.compile) | ~75s | ~75s |
| Warmup (Model-Switch) | ~25s | ~25s |
| Warmup (Folge-Restart) | ~2s | ~2s |
| HTTP-Overhead (ausserhalb Inferenz) | ~0.1s | ~0.1s |
| Voice Prompt Build (einmalig) | — | ~0.76s (danach gecacht) |

**GPU Power State**: Nach ~20s Idle mit `GPU_MAX_HW_QUEUES=1` faellt die GPU in einen niedrigen Takt-Zustand. Der erste Request danach zeigt erhoehte RTF (~0.97x), normalisiert sich nach 2-3 Requests. Das ist kein Bug, sondern Hardware-Verhalten.

### Streaming-Benchmark (decode_window_frames / emit_every_frames)

Beim Streaming wird pro Emit-Zyklus das gesamte Decode-Fenster neu decoded, aber nur `emit_every_frames` neue Audio-Samples extrahiert. Die **Redundancy Ratio** (= `decode_window / emit_every`) bestimmt den Overhead.

**Optimale Config: `decode_window_frames=72, emit_every_frames=24` (Redundancy ~3x)**

Benchmark-Ergebnisse (AMD Radeon 8060S, Strix Halo, mittellange deutsche Saetze):

| decode_window | emit_every | Redundancy | TTFB | RTF (med) | Status |
|---|---|---|---|---|---|
| 48 | 16 | 3.0x | 5.26s | 1.97x | DEFEKT (CUDA graph bug) |
| 64 | 21 | 3.0x | 3.52s | 1.85x | DEFEKT (CUDA graph bug) |
| **72** | **24** | **3.0x** | **1.09s** | **0.57x** | **OPTIMAL** |
| 80 | 8 | 10.0x | 10.7s | 5.5x | DEFEKT (CUDA graph bug) |
| 84 | 28 | 3.0x | 1.30s | 0.58x | OK |
| 84 | 8 | 10.5x | 1.28s | 0.67x | OK (niedriger TTFB, hoeherer RTF) |
| 150 | 50 | 3.0x | 2.24s | 0.58x | OK |
| 200 | 67 | 3.0x | 2.97s | 0.57x | OK |
| 300 | 100 | 3.0x | 4.40s | 0.58x | OK |
| 300 | 150 | 2.0x | 4.93s | 0.55x | OK (bester RTF, aber TTFB zu hoch) |
| 300 | 8 | 37.5x | 1.92s | 0.99x | ALT (unnoetige Redundanz) |

**Erkenntnisse:**

1. **Redundancy ~3x ist optimal.** Bei gleicher Ratio sind RTF-Werte fast identisch (0.57-0.58x), unabhaengig von der absoluten Fenstergroesse.
2. **Kleinere Fenster = niedrigerer TTFB** bei gleichem RTF. 72/24 hat 1.09s TTFB vs. 300/100 mit 4.40s.
3. **Werte ≤64 und genau 80 sind auf AMD/ROCm defekt.** CUDA Graph Capture schlaegt fehl, Fallback auf unkompilierten Pfad (5-10x langsamer). Grenze liegt zwischen 64 und 72.
4. **Non-Streaming ist unabhaengig** von diesen Werten. Dort wird `chunked_decode(chunk_size=300, left_context=25)` mit nur 8% Ueberlappung genutzt — Bottleneck ist die autoregressive Generierung, nicht das Decode.

## Dateien in diesem Ordner

```
install/
  INSTALL.md            -- Diese Anleitung
  config.yaml           -- Server-Konfiguration
  groxaxo/              -- Modifizierte Dateien fuer groxaxo-Repo
    optimized_backend.py
    factory.py
    openai_compatible.py
    schemas.py
    start_server.sh
  dffdeeq/              -- Modifizierte Dateien fuer dffdeeq-Repo
    modeling_qwen3_tts_tokenizer_v2.py
    configuration_qwen3_tts_tokenizer_v2.py
    qwen3_tts_model.py
```

## Installation

### 1. Container erstellen

```bash
set +H
distrobox create --name qwen3-tts \
  --image docker.io/kyuz0/amd-strix-halo-comfyui:latest \
  --additional-flags "--device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined" \
  --init-hooks 'cd /opt/qwen3-tts && ./start_server.sh &'
```

`--init-hooks` fuehrt den angegebenen Befehl bei jedem Container-Start aus. Damit startet der TTS-Server automatisch im Hintergrund.

### 2. Container betreten

```bash
distrobox enter qwen3-tts
```

Alle folgenden Befehle werden **im Container** ausgefuehrt.

### 3. Repos klonen

```bash
sudo mkdir -p /opt/qwen3-tts /opt/qwen3-tts-streaming
sudo chown $(whoami):$(whoami) /opt/qwen3-tts /opt/qwen3-tts-streaming
git clone https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git /opt/qwen3-tts
git clone https://github.com/dffdeeq/Qwen3-TTS-streaming.git /opt/qwen3-tts-streaming
```

### 4. Python-Dependencies installieren

```bash
source /opt/venv/bin/activate
cd /opt/qwen3-tts && pip install -e ".[api]"
cd /opt/qwen3-tts-streaming && pip install -e .
```

### 5. Flash Attention bauen (AMD)

```bash
source /opt/venv/bin/activate
cd /opt
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install . --no-build-isolation
cd /opt && rm -rf flash-attention
python3 -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} OK')"
```

### 6. Symlink erstellen

```bash
cd /opt/qwen3-tts
mv qwen_tts qwen_tts.bak
ln -s /opt/qwen3-tts-streaming/qwen_tts qwen_tts
```

### 7. Modifizierte groxaxo-Dateien kopieren

```bash
cp groxaxo/optimized_backend.py  /opt/qwen3-tts/api/backends/optimized_backend.py
cp groxaxo/factory.py            /opt/qwen3-tts/api/backends/factory.py
cp groxaxo/openai_compatible.py  /opt/qwen3-tts/api/routers/openai_compatible.py
cp groxaxo/schemas.py            /opt/qwen3-tts/api/structures/schemas.py
cp groxaxo/start_server.sh       /opt/qwen3-tts/start_server.sh
chmod +x /opt/qwen3-tts/start_server.sh
```

### 8. Modifizierte dffdeeq-Dateien kopieren

```bash
cp dffdeeq/modeling_qwen3_tts_tokenizer_v2.py      /opt/qwen3-tts-streaming/qwen_tts/core/tokenizer_12hz/
cp dffdeeq/configuration_qwen3_tts_tokenizer_v2.py  /opt/qwen3-tts-streaming/qwen_tts/core/tokenizer_12hz/
cp dffdeeq/qwen3_tts_model.py                       /opt/qwen3-tts-streaming/qwen_tts/inference/
```

### 9. .pyc-Cache loeschen

```bash
find /opt/qwen3-tts /opt/qwen3-tts-streaming -name "*.pyc" -delete
```

### 10. Konfiguration

```bash
mkdir -p ~/qwen3-tts/voice_library
cp config.yaml ~/qwen3-tts/config.yaml
```

Die Modell-Pfade in `config.yaml` zeigen auf `~/qwen3-tts/models/`. Bei anderem Home-Verzeichnis anpassen.

### 11. Modelle herunterladen

```bash
source /opt/venv/bin/activate
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', local_dir='$HOME/qwen3-tts/models/0.6B-CustomVoice')
snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base', local_dir='$HOME/qwen3-tts/models/0.6B-Base')
print('OK')
"
```

### 12. Container verlassen und neustarten

```bash
exit
podman restart qwen3-tts
```

Erster Start: ~75s (torch.compile Warmup). Danach ~2s pro Restart.

## Updates deployen (ohne Neuinstallation)

Aenderungen an den install-Dateien werden per `podman cp` von aussen in den laufenden Container kopiert:

```bash
# groxaxo-Dateien
podman cp install/groxaxo/optimized_backend.py  qwen3-tts:/opt/qwen3-tts/api/backends/optimized_backend.py
podman cp install/groxaxo/openai_compatible.py  qwen3-tts:/opt/qwen3-tts/api/routers/openai_compatible.py
podman cp install/groxaxo/start_server.sh       qwen3-tts:/opt/qwen3-tts/start_server.sh

# dffdeeq-Dateien
podman cp install/dffdeeq/modeling_qwen3_tts_tokenizer_v2.py  qwen3-tts:/opt/qwen3-tts-streaming/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py
podman cp install/dffdeeq/qwen3_tts_model.py                  qwen3-tts:/opt/qwen3-tts-streaming/qwen_tts/inference/qwen3_tts_model.py

# .pyc-Cache loeschen (WICHTIG! Sonst laedt Python alte kompilierte Versionen)
podman exec qwen3-tts find /opt/qwen3-tts /opt/qwen3-tts-streaming -name "*.pyc" -delete

# Server neustarten
podman restart qwen3-tts
```

**Hinweis**: `distrobox enter -- sudo cp` funktioniert NICHT zuverlaessig — nutze immer `podman cp`.

## Streaming-Verhalten

**CustomVoice mit `stream=true`**: Nutzt dffdeeqs `stream_generate_custom_voice()` mit echtem Token-Streaming. WAV-Format wird dabei automatisch auf PCM umgeleitet (WAV braucht Content-Length Header, unvereinbar mit Streaming).

**Voice Cloning (`clone:Name`) mit `stream=true`**: Nutzt `stream_generate_voice_clone()` mit Voice-Prompt-Cache. Erste Anfrage berechnet Speaker-Embedding (~0.76s), danach aus Cache.

**Ohne `stream`**: Generiert das komplette Audio, dann streamt es in 4KB-Chunks an den Client (Fake-Streaming).

### OpenWebUI-Integration

OpenWebUI (Read Aloud / Auto-Playback) sendet Saetze **seriell** als einzelne `POST /v1/audio/speech`-Requests (stream=false, format=mp3). Jeder Satz wird erst gesendet wenn der vorherige fertig ist. Die Audiowiedergabe startet aber schon waehrend die naechsten Saetze noch generiert werden (AudioQueue).

Parallele Verarbeitung auf Server-Seite bringt daher nichts — der Bottleneck ist OpenWebUIs serielle Verarbeitung.

## Persistente Daten

Diese Verzeichnisse liegen im Home und ueberleben eine Container-Loeschung:

```
~/qwen3-tts/config.yaml       -- Konfiguration
~/qwen3-tts/models/            -- Modelle (je ~2.4 GB)
~/qwen3-tts/voice_library/     -- Voice-Profile
```
