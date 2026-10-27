"""Sarvam bulbul TTS over its streaming WebSocket: Indian voices for Hindi and English.

One socket per reply. Sarvam sets the language and voice once per connection, so the reply's
first sentence picks them: any Devanagari -> the Hindi persona, otherwise the English one (the
same speaker, Aparna, in both). Each sentence is sent with a flush, and the server answers every
flush with an event "final", so the reply is complete once the input has ended and every flush
has its final. Audio is linear16: raw 16 kHz 16-bit PCM, the same format as the mic path.
"""
import asyncio
import base64
import json
import logging
import os
import re
from collections.abc import AsyncIterator

import websockets

from app.pipeline.tts.base import StreamingTTS

log = logging.getLogger(__name__)
MODEL = "bulbul:v4-flash"
URL = "wss://api.sarvam.ai/text-to-speech/ws?model={model}&send_completion_event=true"
HINDI_VOICE = "aparna_hi_customer"  # chosen by ear over Roman-script and other voices
ENGLISH_VOICE = "aparna_en_companion"  # the same voice, for callers who speak English
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def voice_for(sentence: str, hindi: str = HINDI_VOICE, english: str = ENGLISH_VOICE) -> tuple[str, str]:
    return (hindi, "hi-IN") if DEVANAGARI.search(sentence) else (english, "en-IN")


def config_message(speaker: str, language: str) -> str:
    return json.dumps({"type": "config", "data": {
        "speaker": speaker, "language_code": language, "speech_sample_rate": 16000,
        "output_audio_codec": "linear16",  # raw PCM, no WAV header (checked live)
        "enable_preprocessing": True,  # Sarvam's advice for mixed-script text
    }})


def text_message(sentence: str) -> str:
    return json.dumps({"type": "text", "data": {"text": sentence.strip()}})


FLUSH = json.dumps({"type": "flush"})


def parse_message(raw: str) -> tuple[bytes | None, bool]:
    """One server message -> (audio chunk or None, is it a flush's final event)."""
    data = json.loads(raw)
    if data.get("type") == "error":
        raise RuntimeError(f"Sarvam TTS: {data.get('data') or data}")
    if data.get("type") == "audio":
        return base64.b64decode(data["data"]["audio"]), False
    return None, data.get("type") == "event" and data.get("data", {}).get("event_type") == "final"


class SarvamTTS(StreamingTTS):
    def __init__(self, hindi_voice: str | None = None, english_voice: str | None = None, model: str = MODEL):
        self.hindi_voice = hindi_voice or os.getenv("SARVAM_TTS_VOICE_HI") or HINDI_VOICE
        self.english_voice = english_voice or os.getenv("SARVAM_TTS_VOICE_EN") or ENGLISH_VOICE
        self.model = model

    async def stream(self, sentences: AsyncIterator[str]) -> AsyncIterator[bytes]:
        first = await anext(sentences, None)
        if first is None:
            return
        speaker, language = voice_for(first, self.hindi_voice, self.english_voice)

        async def all_sentences() -> AsyncIterator[str]:
            yield first
            async for sentence in sentences:
                yield sentence

        headers = {"api-subscription-key": os.environ["SARVAM_API_KEY"]}
        async with websockets.connect(URL.format(model=self.model), additional_headers=headers) as ws:
            await ws.send(config_message(speaker, language))
            state = {"flushes": 0, "finals": 0, "input_done": False}
            sender = asyncio.create_task(self._send(ws, all_sentences(), state))
            try:
                async for raw in ws:
                    audio, final = parse_message(raw)
                    if audio:
                        yield audio
                    if final:
                        state["finals"] += 1
                        if state["input_done"] and state["finals"] >= state["flushes"]:
                            break
            finally:
                sender.cancel()  # consumer stopped early (barge-in) or the reply is complete

    async def _send(self, ws, sentences: AsyncIterator[str], state: dict) -> None:
        try:
            async for sentence in sentences:
                if sentence.strip():
                    await ws.send(text_message(sentence))
                    await ws.send(FLUSH)  # synthesize now, don't wait for more text
                    state["flushes"] += 1
            state["input_done"] = True
            if state["finals"] >= state["flushes"]:
                # Every final arrived before the LLM finished: the server has nothing more to
                # send, so close, or the receive loop would wait for Sarvam's idle timeout.
                await ws.close()
        except Exception as e:
            log.warning("TTS input stopped: %r", e)
            await ws.close()
            raise
