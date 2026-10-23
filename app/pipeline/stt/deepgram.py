"""Deepgram live STT (nova-3) over a raw WebSocket, behind the same interface as Sarvam.

Protocol (checked against a live session):
  auth   -> header "Authorization: Token <key>"
  client -> binary PCM frames; {"type": "Finalize"} / {"type": "KeepAlive"} / {"type": "CloseStream"}
  server -> {"type": "Results", "is_final", "speech_final", "from_finalize",
             "channel": {"alternatives": [{"transcript": "..."}]}}, "Metadata"
Deepgram cuts one utterance into several is_final segments. We collect them and emit one
final Transcript when the reply to our Finalize arrives (from_finalize = true).
"""
import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from urllib.parse import urlencode

import websockets

from app.pipeline.stt.base import StreamingSTT, Transcript

log = logging.getLogger(__name__)

URL = "wss://api.deepgram.com/v1/listen"
CHUNK_BYTES = 3200  # 100 ms per message, same as for Sarvam
KEEPALIVE_S = 5


def build_url(language: str | None = None) -> str:
    params = {
        "model": "nova-3",
        # "hi" = Hindi; "multi" = code-switching across 10 languages including Hindi.
        # On a short quiet clip "multi" guessed Spanish, so "hi" is the default until WER decides.
        "language": language or os.getenv("DEEPGRAM_LANGUAGE") or "hi",
        "encoding": "linear16",
        "sample_rate": 16000,
        "channels": 1,
        "interim_results": "true",
        "punctuate": "true",
    }
    return f"{URL}?{urlencode(params)}"


def parse_results(raw: str) -> tuple[str, bool, bool] | None:
    """Returns (text, is_final, from_finalize) for a Results message, else None."""
    msg = json.loads(raw)
    if msg.get("type") != "Results":
        return None
    text = msg["channel"]["alternatives"][0]["transcript"]
    return text, bool(msg.get("is_final")), bool(msg.get("from_finalize"))


class DeepgramSTT(StreamingSTT):
    def __init__(self, api_key: str | None = None, language: str | None = None):
        self.api_key = api_key or os.environ["DEEPGRAM_API_KEY"]
        self.url = build_url(language)
        self._ws = None
        self._buffer = b""
        self._segments: list[str] = []  # is_final pieces of the current utterance
        self._keepalive = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.url, additional_headers={"Authorization": f"Token {self.api_key}"})
        self._keepalive = asyncio.create_task(self._ping_forever())

    async def _ping_forever(self) -> None:
        while True:
            await asyncio.sleep(KEEPALIVE_S)
            await self._ws.send(json.dumps({"type": "KeepAlive"}))

    async def start_utterance(self) -> None:
        self._buffer = b""
        self._segments = []

    async def send_audio(self, pcm: bytes) -> None:
        self._buffer += pcm
        while len(self._buffer) >= CHUNK_BYTES:
            chunk, self._buffer = self._buffer[:CHUNK_BYTES], self._buffer[CHUNK_BYTES:]
            await self._ws.send(chunk)

    async def end_utterance(self) -> None:
        if self._buffer:
            await self._ws.send(self._buffer)
            self._buffer = b""
        await self._ws.send(json.dumps({"type": "Finalize"}))

    async def transcripts(self) -> AsyncIterator[Transcript]:
        async for raw in self._ws:
            parsed = parse_results(raw)
            if parsed is None:
                continue
            text, is_final, from_finalize = parsed
            if is_final and text:
                self._segments.append(text)
            if from_finalize:
                yield Transcript(" ".join(self._segments), final=True)
                self._segments = []
            elif not is_final and text:
                yield Transcript(" ".join(self._segments + [text]), final=False)

    async def close(self) -> None:
        if self._keepalive:
            self._keepalive.cancel()
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except websockets.ConnectionClosed:
                pass
            await self._ws.close()
