# Pipecat Voice Agent — Installation

## Voraussetzungen
- Server mit Podman (Arch Linux, `--network host`)
- Qwen3-TTS Server läuft (Port 8880)
- llamaswap/llama.cpp läuft (Port 5000)
- Caddy für HTTPS (selbstsignierte Zertifikate)

## 1. Dateien auf den Server kopieren

```bash
scp install/pipecat/start.sh install/pipecat/requirements.txt \
    install/pipecat/app.py install/pipecat/config.yaml \
    pawel@192.168.1.146:~/pipecat/

ssh pawel@192.168.1.146 "sed -i 's/\r$//' ~/pipecat/*.sh ~/pipecat/*.py ~/pipecat/*.txt ~/pipecat/*.yaml"
```

## 2. Container erstellen

```bash
podman run -d --name pipecat \
  --network host \
  -v ~/pipecat:/data:z \
  python:3.13-slim \
  bash /data/start.sh
```

Beim ersten Start:
- Installiert System-Dependencies (libgl1, libglib2.0, libxcb1)
- Erstellt Python venv in `/data/venv/`
- Installiert pip-Pakete aus `requirements.txt`
- Startet die App

## 3. Whisper-Modell vorladen

Beim ersten Start muss das STT-Modell runtergeladen werden (~500 MB):

```bash
podman exec pipecat /data/venv/bin/pip install requests
podman exec pipecat /data/venv/bin/python3 -c "
from faster_whisper import WhisperModel
WhisperModel('small', device='cpu', compute_type='default')
print('Done')
"
podman restart pipecat
```

## 4. Caddy HTTPS-Proxy

In `/etc/caddy/Caddyfile` hinzufügen:

```
:8446 {
    tls /etc/caddy/certs/server.crt /etc/caddy/certs/server.key
    reverse_proxy localhost:7860
}
```

```bash
sudo systemctl reload caddy
```

## 5. Zugriff

**https://192.168.1.146:8446** — Zertifikatswarnung beim ersten Mal akzeptieren.

## Konfiguration

`~/pipecat/config.yaml` auf dem Server editieren, dann `podman restart pipecat`.

```yaml
llm:
  base_url: "http://localhost:5000/v1"
  model: "GPT-OSS (120b-MXFP4)"        # beliebiges llamaswap-Modell
  api_key: "not-needed"

tts:
  base_url: "http://localhost:8880/v1"
  voice: "clone:friedrich"               # oder "alloy" etc.
  model: "tts-1"
  sample_rate: 24000
  api_key: "not-needed"

stt:
  model: "small"                         # tiny/small/medium/large-v3
  device: "cpu"
  language: "de"                         # oder "en", "auto"

agent:
  system_prompt: "..."
  vad_min_volume: 0.5
  vad_stop_secs: 0.8
  allow_interruptions: true
```

## Dateien

| Datei | Zweck |
|-------|-------|
| `app.py` | Bot-Pipeline (Pipecat built-in runner) |
| `config.yaml` | Alle Einstellungen (LLM, TTS, STT) |
| `requirements.txt` | Python-Dependencies |
| `start.sh` | Container-Startscript (venv, deps, app) |

## Logs

```bash
podman logs -f pipecat
```

## Bekannte Eigenheiten
- `VALID_VOICES`-Hack in app.py: Pipecat validiert Voice-Namen gegen OpenAI-Liste, custom Namen wie `clone:xyz` müssen manuell registriert werden
- Pipecat erwartet `response_format: pcm` vom TTS-Server (kein MP3)
- Whisper läuft auf CPU (GPU bleibt frei für TTS)
- Container-Stopp dauert ~10s (SIGTERM → SIGKILL)
