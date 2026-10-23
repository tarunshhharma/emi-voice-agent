import asyncio

from app.pipeline.session import PREROLL_FRAMES, VoiceSession
from app.pipeline.stt.base import Transcript


class FakeVAD:
    """Speaks during the frame indexes given; reports start/end exactly at the edges."""

    def __init__(self, speech_frames: range):
        self.speech = speech_frames
        self.speaking = False
        self.i = 0

    def feed(self, frame):
        now = self.i in self.speech
        events = []
        if now and not self.speaking:
            events.append("speech_start")
        if self.speaking and not now:
            events.append("speech_end")
        self.speaking = now
        self.i += 1
        return events


class FakeSTT:
    def __init__(self, transcripts=()):
        self.log = []  # every call, in order
        self._transcripts = list(transcripts)

    async def start_utterance(self):
        self.log.append("start")

    async def send_audio(self, pcm):
        self.log.append(pcm)

    async def end_utterance(self):
        self.log.append("end")

    async def transcripts(self):
        for t in self._transcripts:
            yield t


def frame(i: int) -> bytes:
    return i.to_bytes(2, "little") * 320  # a different, recognisable 640-byte frame per index


def run_frames(vad, stt, n):
    sent = []

    async def send_json(m):
        sent.append(m)

    async def go():
        session = VoiceSession(vad, stt, send_json)
        for i in range(n):
            await session.on_frame(frame(i))
        return session

    session = asyncio.run(go())
    return session, sent


def test_only_speech_and_preroll_reach_stt():
    stt = FakeSTT()
    run_frames(FakeVAD(range(40, 60)), stt, 100)
    audio = [x for x in stt.log if isinstance(x, bytes)]
    # 300 ms of pre-roll before the start, then the 20 speech frames; nothing after.
    expected = [frame(i) for i in range(40 - PREROLL_FRAMES, 40)] + [frame(i) for i in range(40, 60)]
    assert audio == expected


def test_stt_calls_are_in_order():
    stt = FakeSTT()
    run_frames(FakeVAD(range(10, 20)), stt, 30)
    markers = [x for x in stt.log if isinstance(x, str)]
    assert markers == ["start", "end"]
    assert stt.log[0] == "start" and stt.log[-1] == "end"


def test_browser_is_told_when_speech_starts_and_stops():
    _, sent = run_frames(FakeVAD(range(5, 10)), FakeSTT(), 20)
    assert sent == [{"type": "vad", "speaking": True}, {"type": "vad", "speaking": False}]


def test_final_transcript_carries_stt_latency():
    stt = FakeSTT([Transcript("ha", False), Transcript("haan ji", True)])
    session, _ = run_frames(FakeVAD(range(2, 5)), stt, 10)  # speech ended, so a timer is running
    sent = []

    async def send_json(m):
        sent.append(m)

    session.send_json = send_json
    asyncio.run(session.forward_transcripts())
    assert sent[0] == {"type": "transcript", "text": "ha", "final": False}
    assert sent[1]["final"] and sent[1]["text"] == "haan ji"
    assert sent[1]["stt_ms"] >= 0
