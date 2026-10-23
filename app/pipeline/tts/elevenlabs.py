"""ElevenLabs streaming TTS over the stream-input WebSocket.

One WebSocket per agent reply: open, send each sentence as it arrives (flush, so audio
starts without waiting for more text), close with an empty text. Audio comes back as
base64 PCM chunks at 16 kHz, the same format as the mic path.
"""
import asyncio
import base64
import json
import logging
import os
from collections.abc import AsyncIterator

import websockets

from app.pipeline.tts.base import StreamingTTS

log = logging.getLogger(__name__)
MODEL = "eleven_flash_v2_5"  # ~75 ms model latency per the docs; supports Hindi
DEFAULT_VOICE = "EXAVITQu4vr4xnSDxMaL"  # "Sarah", a premade voice; override with ELEVENLABS_VOICE_ID
URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice}/stream-input?model_id={model}&output_format=pcm_16000"


def text_message(sentence: str) -> str:
    # The API wants every text chunk to end with one space. flush=True asks for audio
    # now instead of waiting to fill its buffer, which matters when sentences trickle in.
    return json.dumps({"text": sentence.strip() + " ", "flush": True})


def parse_message(raw: str) -> tuple[bytes | None, bool]:
    """One server message -> (audio chunk or None, is this the last message)."""
    data = json.loads(raw)
    if "error" in data:
        raise RuntimeError(f"ElevenLabs: {data.get('message') or data['error']}")
    audio = base64.b64decode(data["audio"]) if data.get("audio") else None
    return audio, bool(data.get("isFinal"))


class ElevenLabsTTS(StreamingTTS):
    def __init__(self, voice_id: str | None = None, model: str = MODEL):
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE
        self.model = model

    async def stream(self, sentences: AsyncIterator[str]) -> AsyncIterator[bytes]:
        url = URL.format(voice=self.voice_id, model=self.model)
        headers = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}
        async with websockets.connect(url, additional_headers=headers) as ws:
            await ws.send(json.dumps({"text": " "}))  # required first message
            # Sending and receiving overlap: audio for sentence 1 plays while sentence 2 is sent.
            sender = asyncio.create_task(self._send(ws, sentences))
            try:
                async for raw in ws:
                    audio, final = parse_message(raw)
                    if audio:
                        yield audio
                    if final:
                        break
            finally:
                sender.cancel()  # consumer stopped early (barge-in) or the stream ended

    async def _send(self, ws, sentences: AsyncIterator[str]) -> None:
        try:
            async for sentence in sentences:
                if sentence.strip():
                    await ws.send(text_message(sentence))
            await ws.send(json.dumps({"text": ""}))  # end of input: server finishes, then sends isFinal
        except Exception as e:
            log.warning("TTS input stopped: %r", e)
            await ws.close()  # otherwise the receive loop waits for the 20 s inactivity timeout
            raise
