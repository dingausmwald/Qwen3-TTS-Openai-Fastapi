"""Endpoints for preparing voice-clone reference recordings.

Both are helpers for the two UIs; neither is part of the OpenAI-compatible
surface, so they live in their own router.
"""
import logging

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["reference"])


@router.post("/enhance")
async def enhance_reference(file: UploadFile = File(...)):
    """Restore a noisy reference recording with Sidon. Returns 48 kHz mono WAV."""
    import anyio

    from api.services import sidon_enhance

    data = await file.read()
    if not data:
        return JSONResponse(status_code=400, content={"error": "Leere Datei."})
    try:
        # CPU bound and several seconds long: keep the event loop responsive
        out = await anyio.to_thread.run_sync(sidon_enhance.restore, data)
    except Exception as e:
        logger.exception("Sidon enhance failed")
        return JSONResponse(status_code=500, content={"error": "Verbesserung fehlgeschlagen: %s" % e})

    return Response(
        content=out,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="enhanced.wav"'},
    )


@router.post("/transcribe-reference")
async def transcribe_reference(file: UploadFile = File(...)):
    """Transcribe a reference recording so the transcript field can be prefilled."""
    import anyio

    from api.services import stt_client

    data = await file.read()
    if not data:
        return JSONResponse(status_code=400, content={"error": "Leere Datei."})

    text, err = await anyio.to_thread.run_sync(
        stt_client.transcribe, data, file.filename or "ref.wav"
    )
    if err:
        # 200 with an error field: the caller shows it in the UI rather than failing
        return JSONResponse(content={"text": "", "error": err})
    return JSONResponse(content={"text": text, "error": ""})
