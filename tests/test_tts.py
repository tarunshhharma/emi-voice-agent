import asyncio
import base64
import json

import pytest

from app.pipeline.session import VoiceSession, from_list
from app.pipeline.tts.elevenlabs import parse_message, text_message


def test_text_message_ends_with_one_space_and_flushes():
    m = json.loads(text_message("  Namaste ji.  "))
    assert m == {"text": "Namaste ji. ", "flush": True}


def test_parse_audio_chunk():
    pcm = b"\x01\x00\xff\x7f"
    audio, final = parse_message(json.dumps({"audio": base64.b64encode(pcm).decode(), "isFinal": None}))
    assert audio == pcm and final is False


def test_parse_final_message_has_no_audio():
    assert parse_message(json.dumps({"audio": None, "isFinal": True})) == (None, True)


def test_parse_error_raises():
    with pytest.raises(RuntimeError, match="invalid voice"):
        parse_message(json.dumps({"error": "bad_request", "message": "invalid voice"}))


class FakeTTS:
    def __init__(self, fail=False):
        self.heard = []
        self.fail = fail

    async def stream(self, sentences):
        async for s in sentences:
            self.heard.append(s)
            if self.fail:
                raise RuntimeError("no key")
            yield s.encode()  # one fake "audio" chunk per sentence


def make_session(tts):
    sent, audio = [], []

    async def send_json(m):
        sent.append(m)

    async def send_audio(b):
        audio.append(b)

    return VoiceSession(None, None, send_json, tts=tts, send_audio=send_audio), sent, audio


def test_speak_sends_each_sentence_then_audio_and_first_audio_time():
    tts = FakeTTS()
    session, sent, audio = make_session(tts)
    asyncio.run(session.speak(from_list(["Namaste.", "Kya main Rahul ji se baat kar rahi hoon?"])))
    assert tts.heard == ["Namaste.", "Kya main Rahul ji se baat kar rahi hoon?"]
    texts = [m for m in sent if m["type"] == "agent_text"]
    assert [(m["text"], m["first"]) for m in texts] == [
        ("Namaste.", True), ("Kya main Rahul ji se baat kar rahi hoon?", False)]
    assert len([m for m in sent if m["type"] == "tts"]) == 1  # timed once, on the first chunk
    assert audio == [b"Namaste.", b"Kya main Rahul ji se baat kar rahi hoon?"]


def test_speak_with_nothing_to_say_opens_no_tts_stream():
    tts = FakeTTS()
    session, sent, audio = make_session(tts)
    asyncio.run(session.speak(from_list([])))
    assert sent == [] and audio == [] and tts.heard == []


def test_speak_reports_tts_failure_instead_of_crashing():
    session, sent, audio = make_session(FakeTTS(fail=True))
    asyncio.run(session.speak(from_list(["Namaste."])))
    assert sent[-1]["type"] == "error" and "no key" in sent[-1]["message"]
    assert audio == []
