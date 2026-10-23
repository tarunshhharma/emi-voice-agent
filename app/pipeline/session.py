"""One voice session per call: frames in -> VAD -> STT -> transcripts out.

For now it ends at transcripts. Day 4 adds the agent reply (LLM + TTS) and barge-in here.
"""
import time
from collections import deque

from app.pipeline.stt.base import StreamingSTT

PREROLL_FRAMES = 15  # 300 ms kept from before speech was detected, so the first word isn't clipped


class VoiceSession:
    def __init__(self, vad, stt: StreamingSTT, send_json):
        self.vad = vad
        self.stt = stt
        self.send_json = send_json  # async function that sends a message to the caller's browser
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
