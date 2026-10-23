"""Sarvam realtime STT (saaras:v3-realtime) over a raw WebSocket.

Protocol (checked against a live session):
  client -> {"event": "speech_start"} / {"event": "audio_input", "audio": "<base64 PCM>"} /
            {"event": "speech_end"} / {"event": "ping"} / {"event": "end"}
  server -> {"event": "transcript.partial" | "transcript.final", "text": "..."}, "pong",
            "session.begin", "session.end", "error"
We use manual endpointing: our own Silero VAD decides when a turn starts and ends.
"""
import asyncio
import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from urllib.parse import urlencode

import websockets

from app.pipeline.stt.base import StreamingSTT, Transcript

log = logging.getLogger(__name__)

URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
CHUNK_BYTES = 3200  # 100 ms of audio per message, as in Sarvam's examples (fewer, larger messages)
KEEPALIVE_S = 5  # we only send audio while someone speaks, so keep the line alive in between


def build_url(language_code: str = "hi-IN", mode: str | None = None) -> str:
    params = {
        "language_code": language_code,
        "model": "saaras:v3-realtime",
        "endpointing": "manual",
        "encoding": "linear16",
        "sample_rate": 16000,
        # translit = Roman script ("Rahul Sharma"), which matches how names are stored for
        # verification. Other modes return Devanagari. Compare accuracy in the WER test.
        "mode": mode or os.getenv("SARVAM_STT_MODE") or "translit",
    }
    return f"{URL}?{urlencode(params)}"


def parse_message(raw: str) -> Transcript | None:
    msg = json.loads(raw)
    event = msg.get("event")
    if event == "error":
        log.warning("sarvam error: %s", msg)
    if event in ("transcript.partial", "transcript.final"):
        return Transcript(text=msg.get("text", ""), final=event == "transcript.final")
    return None


class SarvamSTT(StreamingSTT):
    def __init__(self, api_key: str | None = None, language_code: str = "hi-IN"):
        self.api_key = api_key or os.environ["SARVAM_API_KEY"]
        self.url = build_url(language_code)
        self._ws = None
        self._buffer = b""
        self._keepalive = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.url, additional_headers={"api-subscription-key": self.api_key})
        self._keepalive = asyncio.create_task(self._ping_forever())

    async def _send(self, event: dict) -> None:
        await self._ws.send(json.dumps(event))

    async def _ping_forever(self) -> None:
        while True:
            await asyncio.sleep(KEEPALIVE_S)
            await self._send({"event": "ping"})

    async def start_utterance(self) -> None:
        self._buffer = b""
        await self._send({"event": "speech_start"})

    async def send_audio(self, pcm: bytes) -> None:
        self._buffer += pcm
        while len(self._buffer) >= CHUNK_BYTES:
            chunk, self._buffer = self._buffer[:CHUNK_BYTES], self._buffer[CHUNK_BYTES:]
            await self._send({"event": "audio_input", "audio": base64.b64encode(chunk).decode()})

    async def end_utterance(self) -> None:
        if self._buffer:  # the last partial chunk
            await self._send({"event": "audio_input", "audio": base64.b64encode(self._buffer).decode()})
            self._buffer = b""
        await self._send({"event": "speech_end"})

    async def transcripts(self) -> AsyncIterator[Transcript]:
        async for raw in self._ws:
            transcript = parse_message(raw)
            if transcript:
                yield transcript

    async def close(self) -> None:
        if self._keepalive:
            self._keepalive.cancel()
        if self._ws:
            try:
                await self._send({"event": "end"})
            except websockets.ConnectionClosed:
                pass
            await self._ws.close()
