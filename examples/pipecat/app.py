"""Pipecat Voice Agent — Qwen3-TTS + faster-whisper + LLM

A real-time voice agent that combines:
- STT: faster-whisper (local, CPU)
- LLM: any OpenAI-compatible API (llamaswap, ollama, etc.)
- TTS: Qwen3-TTS with optional real-time streaming

See config.yaml for all settings.
"""

import yaml
from typing import AsyncGenerator
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    ErrorFrame, Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService, VALID_VOICES
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport


class StreamingOpenAITTSService(OpenAITTSService):
    """OpenAI TTS that sends stream=true for real-time chunk delivery."""

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS (streaming) [{text}]")
        try:
            from openai import BadRequestError
            await self.start_ttfb_metrics()

            async with self._client.audio.speech.with_streaming_response.create(
                input=text,
                model=self.model_name,
                voice=self._voice_id,
                response_format="pcm",
                extra_body={"stream": True},
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(f"{self} error (status: {r.status_code}, error: {error})")
                    yield ErrorFrame(error=f"Error getting audio (status: {r.status_code})")
                    return

                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()
                async for chunk in r.iter_bytes(self.chunk_size):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
                yield TTSStoppedFrame()
        except BadRequestError as e:
            yield ErrorFrame(error=f"TTS error: {e}")


def _create_tts_service(cfg: dict) -> OpenAITTSService:
    """Create TTS service based on config: streaming or non-streaming."""
    streaming = cfg["tts"].get("streaming", False)
    cls = StreamingOpenAITTSService if streaming else OpenAITTSService
    logger.info(f"TTS mode: {'streaming' if streaming else 'non-streaming'}")
    return cls(
        api_key=cfg["tts"]["api_key"],
        base_url=cfg["tts"]["base_url"],
        voice=cfg["tts"]["voice"],
        model=cfg["tts"]["model"],
        sample_rate=cfg["tts"]["sample_rate"],
    )

# ---------------------------------------------------------------------------
# Load config from config.yaml
# ---------------------------------------------------------------------------
with open("/data/config.yaml") as f:
    cfg = yaml.safe_load(f)

logger.info(f"Config loaded: LLM={cfg['llm']['model']}, TTS voice={cfg['tts']['voice']}")

# Allow any voice name for custom TTS servers (Qwen3-TTS uses clone:name etc.)
voice_name = cfg["tts"]["voice"]
if voice_name not in VALID_VOICES:
    VALID_VOICES[voice_name] = voice_name


# ---------------------------------------------------------------------------
# Bot entry point
# ---------------------------------------------------------------------------
async def bot(args: SmallWebRTCRunnerArguments):
    transport = SmallWebRTCTransport(
        webrtc_connection=args.webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    min_volume=cfg["agent"]["vad_min_volume"],
                    stop_secs=cfg["agent"]["vad_stop_secs"],
                )
            ),
        ),
    )

    stt = WhisperSTTService(
        model=cfg["stt"]["model"],
        device=cfg["stt"]["device"],
        no_speech_prob=0.4,
        language=cfg["stt"]["language"],
    )

    llm = OpenAILLMService(
        api_key=cfg["llm"]["api_key"],
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
    )

    tts = _create_tts_service(cfg)

    messages = [
        {"role": "system", "content": cfg["agent"]["system_prompt"]},
    ]
    context = OpenAILLMContext(messages=messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=cfg["agent"]["allow_interruptions"]),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.queue_frames([])

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
