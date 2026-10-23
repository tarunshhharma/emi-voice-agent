"""One voice session per call: frames in -> VAD -> STT -> transcripts out, and agent text -> TTS -> audio out.

Day 4 Task 2 connects the two: STT final -> LLM -> speak(). Barge-in comes after that.
"""
import logging
import time
from collections import deque

from app.pipeline.stt.base import StreamingSTT
from app.pipeline.tts.base import StreamingTTS

log = logging.getLogger(__name__)
PREROLL_FRAMES = 15  # 300 ms kept from before speech was detected, so the first word isn't clipped


class VoiceSession:
    def __init__(self, vad, stt: StreamingSTT, send_json, tts: StreamingTTS | None = None, send_audio=None):
        self.vad = vad
        self.stt = stt
        self.send_json = send_json  # async function that sends a message to the caller's browser
        self.tts = tts
        self.send_audio = send_audio  # async function that sends PCM bytes to the browser for playback
        self.preroll: deque[bytes] = deque(maxlen=PREROLL_FRAMES)
        self.speech_ended_at: float | None = None

    async def on_frame(self, frame: bytes) -> None:
        for event in self.vad.feed(frame):
            if event == "speech_start":
                await self.stt.start_utterance()
                # VAD needs a little audio before it is sure, so it fires slightly late.
                # Send the frames from just before, so the first syllable reaches STT.
                for earlier in self.preroll:
                    await self.stt.send_audio(earlier)
                self.preroll.clear()
                await self.send_json({"type": "vad", "speaking": True})
            elif event == "speech_end":
                await self.stt.end_utterance()
                self.speech_ended_at = time.monotonic()
                await self.send_json({"type": "vad", "speaking": False})

        if self.vad.speaking:
            await self.stt.send_audio(frame)  # only speech goes to STT: less cost, less noise
        else:
            self.preroll.append(frame)

    async def forward_transcripts(self) -> None:
        """Runs as its own task for the whole call, alongside the frame loop."""
        async for transcript in self.stt.transcripts():
            message = {"type": "transcript", "text": transcript.text, "final": transcript.final}
            if transcript.final and self.speech_ended_at is not None:
                # STT part of the latency budget: our end-of-speech -> provider's final text.
                message["stt_ms"] = round((time.monotonic() - self.speech_ended_at) * 1000)
                self.speech_ended_at = None
            await self.send_json(message)

    async def speak(self, sentences: list[str]) -> None:
        """Say the agent's sentences: text to the browser for display, audio as it is synthesized."""

        async def feed():
            for sentence in sentences:
                yield sentence

        await self.send_json({"type": "agent_text", "text": " ".join(sentences)})
        started = time.monotonic()
        first_audio_ms = None
        try:
            async for chunk in self.tts.stream(feed()):
                if first_audio_ms is None:
                    # TTS part of the latency budget: text sent -> first audio byte back.
                    first_audio_ms = round((time.monotonic() - started) * 1000)
                    log.info("tts first audio %d ms", first_audio_ms)
                    await self.send_json({"type": "tts", "first_audio_ms": first_audio_ms})
                await self.send_audio(chunk)
        except Exception as e:  # bad key, network down: the caller sees the text and an error
            log.warning("TTS failed: %r", e)
            await self.send_json({"type": "error", "message": f"Text-to-speech unavailable: {e}"})
