"""The public demo's call limits, driven through the real WebSocket endpoint with a fake socket."""
import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from app import demo_gate, main


class FakeSocket:
    """Just enough of a FastAPI WebSocket: records what the server sends, never receives a frame."""

    def __init__(self):
        self.sent, self.closed = [], False
        self.query_params = {}
        self.app = SimpleNamespace(state=SimpleNamespace(openai=SimpleNamespace(models=SimpleNamespace(list=lambda: None))))

    async def accept(self):
        pass

    async def send_json(self, message):
        self.sent.append(message)

    async def send_bytes(self, data):
        pass

    async def close(self):
        self.closed = True

    async def receive(self):
        await asyncio.Event().wait()  # a caller who keeps the call open and never hangs up


@pytest.fixture(autouse=True)
def no_limits(monkeypatch):
    for name in ("DEMO_ACCESS_CODE", "MAX_CALLS_PER_DAY", "MAX_ACTIVE_CALLS", "MAX_CALL_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(main, "active_calls", 0)


def test_busy():
    assert demo_gate.busy(5, None) is None  # no limit configured
    assert demo_gate.busy(1, 2) is None
    assert "busy" in demo_gate.busy(2, 2)


def test_env_int(monkeypatch):
    assert demo_gate.env_int("MAX_ACTIVE_CALLS") is None
    monkeypatch.setenv("MAX_ACTIVE_CALLS", "2")
    assert demo_gate.env_int("MAX_ACTIVE_CALLS") == 2


def test_refuses_a_call_over_the_active_limit(monkeypatch):
    monkeypatch.setenv("MAX_ACTIVE_CALLS", "2")
    monkeypatch.setattr(main, "active_calls", 2)
    ws = FakeSocket()
    asyncio.run(main.browser_audio(ws))
    assert "busy" in ws.sent[0]["message"] and ws.closed
    assert main.active_calls == 2  # a refused call never takes a slot


def test_a_crashed_call_frees_its_slot(monkeypatch):
    async def crash(ws, db):
        raise RuntimeError("STT went away")
    monkeypatch.setattr(main, "_run_call", crash)
    with pytest.raises(RuntimeError):
        asyncio.run(main.browser_audio(FakeSocket()))
    assert main.active_calls == 0


def test_call_is_ended_at_the_time_limit(monkeypatch):
    closed = []
    agent = SimpleNamespace(off_hours_bypassed=False, close=lambda: closed.append(True),
                            borrower=SimpleNamespace(name="Test", dob=date(1990, 1, 1), phone="9000000000"))
    stt = SimpleNamespace(connect=lambda: asyncio.sleep(0), close=lambda: asyncio.sleep(0))
    session = SimpleNamespace(forward_transcripts=lambda: asyncio.Event().wait(),
                              converse=lambda: asyncio.Event().wait())
    monkeypatch.setattr(main, "CallAgent", lambda *args, **kwargs: agent)
    monkeypatch.setattr(main, "make_stt", lambda: stt)
    monkeypatch.setattr(main, "make_tts", lambda: None)
    monkeypatch.setattr(main.vad, "SileroVAD", lambda: None)
    monkeypatch.setattr(main, "VoiceSession", lambda *args, **kwargs: session)
    monkeypatch.setenv("MAX_CALL_SECONDS", "0")  # fires at the first wait, so the test is instant

    ws = FakeSocket()
    asyncio.run(main.browser_audio(ws))
    assert "time limit" in ws.sent[-1]["message"] and ws.closed
    assert closed and main.active_calls == 0  # the call was closed in the DB and its slot freed
