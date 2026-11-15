"""Transcribe a reference recording via the Parakeet STT service.

Used to prefill the reference transcript when a recording is uploaded. In ICL
mode a wrong transcript is worse than none, so the result is always offered for
editing rather than applied silently, and every failure path returns a message
instead of raising - a stopped STT container must not break the clone tab.
"""
import logging
import os

import requests

logger = logging.getLogger(__name__)

STT_URL = os.getenv("STT_URL", "http://127.0.0.1:5092/v1/audio/transcriptions")
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
STT_TIMEOUT_S = float(os.getenv("STT_TIMEOUT_S", "120"))


def transcribe(audio_bytes: bytes, filename: str = "ref.wav") -> tuple[str, str]:
    """Return (text, error). Exactly one of the two is non-empty."""
    try:
        r = requests.post(
            STT_URL,
            files={"file": (filename, audio_bytes, "application/octet-stream")},
            data={"model": STT_MODEL, "response_format": "json"},
            timeout=STT_TIMEOUT_S,
        )
    except requests.exceptions.ConnectionError:
        return "", "STT-Dienst nicht erreichbar (%s). Container gestartet?" % STT_URL
    except requests.exceptions.Timeout:
        return "", "STT-Dienst hat nicht rechtzeitig geantwortet."
    except Exception as e:
        return "", "STT-Aufruf fehlgeschlagen: %s" % e

    if r.status_code != 200:
        return "", "STT-Dienst antwortete mit %s: %s" % (r.status_code, r.text[:200])

    try:
        text = (r.json().get("text") or "").strip()
    except ValueError:
        text = r.text.strip()

    if not text:
        return "", "STT lieferte keinen Text zurueck."
    return text, ""
