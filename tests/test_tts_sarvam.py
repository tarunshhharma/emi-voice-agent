import asyncio
import base64
import json

import pytest

from app.pipeline.tts import sarvam
from app.pipeline.tts.sarvam import SarvamTTS, config_message, parse_message, text_message, voice_for


def test_the_first_sentence_picks_hindi_or_english_aparna():
    assert voice_for("नमस्ते जी, आपकी EMI पच्चीस हज़ार रुपये है।") == ("aparna_hi_customer", "hi-IN")
    assert voice_for("Thank you, your EMI is due on Friday.") == ("aparna_en_companion", "en-IN")


def test_messages():
    cfg = json.loads(config_message("aparna_hi_customer", "hi-IN"))["data"]
    assert cfg["output_audio_codec"] == "linear16" and cfg["speech_sample_rate"] == 16000
    assert json.loads(text_message("  नमस्ते।  ")) == {"type": "text", "data": {"text": "नमस्ते।"}}
    audio = base64.b64encode(b"\x01\x00").decode()
    assert parse_message(json.dumps({"type": "audio", "data": {"audio": audio}})) == (b"\x01\x00", False)
    assert parse_message(json.dumps({"type": "event", "data": {"event_type": "final"}})) == (None, True)
    with pytest.raises(RuntimeError):
        parse_message(json.dumps({"type": "error", "data": {"message": "bad key"}}))


class FakeSarvam:
    """Answers every flush with one audio chunk and a final, like the real server."""

    def __init__(self):
        self.sent, self.closed = [], False
        self.inbox: asyncio.Queue = asyncio.Queue()

    async def send(self, message):
        self.sent.append(json.loads(message))
        if self.sent[-1]["type"] == "flush":
            text = self.sent[-2]["data"]["text"]
            await self.inbox.put(json.dumps({"type": "audio", "data": {"audio": base64.b64encode(text.encode()).decode()}}))
            await self.inbox.put(json.dumps({"type": "event", "data": {"event_type": "final"}}))

    async def close(self):
        self.closed = True
        await self.inbox.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.inbox.get()
        if message is None:
            raise StopAsyncIteration
        return message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def fake(monkeypatch):
    server = FakeSarvam()
    monkeypatch.setattr(sarvam.websockets, "connect", lambda url, additional_headers: server)
    monkeypatch.setenv("SARVAM_API_KEY", "test")
    return server


def collect(sentence_source):
    async def run():
        return [chunk async for chunk in SarvamTTS().stream(sentence_source())]
    return asyncio.run(asyncio.wait_for(run(), 3))  # a hang (missed end of reply) fails the test


def test_a_reply_ends_when_every_flush_has_its_final(fake):
    async def sentences():
        yield "नमस्ते जी।"
        yield "आपकी EMI पच्चीस हज़ार है।"
    audio = collect(sentences)
    assert [a.decode() for a in audio] == ["नमस्ते जी।", "आपकी EMI पच्चीस हज़ार है।"]
    assert fake.sent[0]["data"]["speaker"] == "aparna_hi_customer"
    assert [m["type"] for m in fake.sent[1:]] == ["text", "flush", "text", "flush"]


def test_no_hang_when_the_last_final_arrives_before_the_llm_finishes(fake):
    async def sentences():
        yield "Thank you."
        await asyncio.sleep(0.2)  # the LLM is still "thinking": all finals are already in
    audio = collect(sentences)  # must end at once, not wait for Sarvam's idle timeout
    assert [a.decode() for a in audio] == ["Thank you."] and fake.closed
    assert fake.sent[0]["data"]["speaker"] == "aparna_en_companion"


def test_nothing_to_say_opens_no_connection(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("should not connect")
    monkeypatch.setattr(sarvam.websockets, "connect", refuse)

    async def nothing():
        return
        yield
    assert collect(nothing) == []
