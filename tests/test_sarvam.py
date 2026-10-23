import json
from urllib.parse import parse_qs, urlparse

from app.pipeline.stt.base import Transcript
from app.pipeline.stt.sarvam import build_url, parse_message


def test_url_uses_manual_endpointing_and_our_audio_format():
    query = parse_qs(urlparse(build_url(mode="translit")).query)
    assert query["endpointing"] == ["manual"]  # our VAD owns turn detection
    assert query["encoding"] == ["linear16"] and query["sample_rate"] == ["16000"]
    assert query["model"] == ["saaras:v3-realtime"] and query["mode"] == ["translit"]


# Message shapes below are copied from a live Sarvam session.

def test_parse_partial():
    raw = json.dumps({"event": "transcript.partial", "utterance_idx": 0, "text": "ओला हो"})
    assert parse_message(raw) == Transcript(text="ओला हो", final=False)


def test_parse_final():
    raw = json.dumps({"event": "transcript.final", "utterance_idx": 0, "text": "Allow."})
    assert parse_message(raw) == Transcript(text="Allow.", final=True)


def test_other_events_are_ignored():
    for event in ("session.begin", "pong", "session.end", "vad.speech_start"):
        assert parse_message(json.dumps({"event": event})) is None


def test_error_is_ignored_but_not_fatal():
    assert parse_message(json.dumps({"event": "error", "code": 4000, "message": "bad"})) is None
