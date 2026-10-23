import asyncio

from app.db.models import Call, Turn
from app.pipeline.call import CallAgent
from app.pipeline.metrics import TurnTimer, percentile, summary
from app.pipeline.session import VoiceSession
from tests.fakes import FakeClient
from tests.test_conversation import FakeAgent, FakeTTS, SlowAgent, voice_session, FRAME, BARGE_IN_FRAMES


def test_durations_split_voice_to_voice_into_stages():
    t = TurnTimer(caller_stopped=10.0, speech_end=10.5, stt_final=10.8,
                  agent_start=10.8, first_sentence=12.3, first_audio=12.7)
    d = t.durations()
    assert d == {"endpoint_ms": 500, "stt_ms": 300, "llm_ttft_ms": 1500, "tts_first_byte_ms": 400,
                 "voice_to_voice_ms": 2700, "barge_in_ms": None}
    stages = d["endpoint_ms"] + d["stt_ms"] + d["llm_ttft_ms"] + d["tts_first_byte_ms"]
    assert stages == d["voice_to_voice_ms"]  # no hidden time when nothing waits in the queue


def test_missing_stage_gives_none_not_a_crash():
    assert TurnTimer(caller_stopped=1.0).durations()["voice_to_voice_ms"] is None


def test_percentile_nearest_rank():
    values = list(range(1, 21))  # 1..20
    assert percentile(values, 50) == 10
    assert percentile(values, 95) == 19
    assert percentile([7], 95) == 7


def test_each_turn_is_saved_with_its_timings():
    agent = FakeAgent({"haan": ["Dhanyavaad.", "DOB bataiye?"]})
    sent = []

    async def send_json(m):
        sent.append(m)

    async def send_audio(b):
        pass

    session = VoiceSession(None, None, send_json, tts=FakeTTS(), send_audio=send_audio, agent=agent)

    async def run():
        timer = TurnTimer(caller_stopped=0.0, speech_end=0.0, stt_final=0.0)
        conversation = asyncio.create_task(session.converse())
        session.user_turns.put_nowait(("haan", timer))
        await asyncio.sleep(0.3)
        conversation.cancel()

    asyncio.run(run())
    opening, reply = agent.saved
    assert opening[0] is None and opening[1] == agent.first_line
    assert reply[:2] == ("haan", "Dhanyavaad. DOB bataiye?") and reply[3] is False
    for stage in ["llm_ttft_ms", "tts_first_byte_ms", "voice_to_voice_ms"]:
        assert reply[2][stage] is not None
    assert [m["type"] for m in sent].count("latency") == 2


def test_an_interrupted_reply_is_saved_with_its_barge_in_time():
    agent = SlowAgent()
    session, sent = voice_session(agent, range(0, 2 * BARGE_IN_FRAMES))

    async def run():
        conversation = asyncio.create_task(session.converse())
        session.user_turns.put_nowait(("haan", TurnTimer()))
        while not agent.started.is_set():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)  # past the Windows clock tick (see the barge-in test)
        for _ in range(2 * BARGE_IN_FRAMES):
            await session.on_frame(FRAME)
        agent.release.set()
        await asyncio.sleep(0.3)
        conversation.cancel()

    asyncio.run(run())
    caller_text, agent_text, timings, interrupted = agent.saved[1]
    assert interrupted is True and timings["barge_in_ms"] is not None
    assert agent_text == "Aapka EMI teen hazaar paanch sau rupaye hai,"  # only what was sent to TTS


def test_saved_rows_feed_the_summary(session):
    agent = CallAgent(session, FakeClient([]), loan_id=1, allow_off_hours=True)
    for v2v in [1200, 1500, 3000]:
        agent.save_turn("haan", "Ji.", {"stt_ms": 300, "voice_to_voice_ms": v2v}, interrupted=False)
    rows = session.query(Turn).filter_by(call_id=agent.call.id).all()
    assert [r.speaker for r in rows] == ["caller", "agent"] * 3
    stats = summary(session, channel="browser")
    assert stats["voice_to_voice_ms"] == {"n": 3, "p50": 1500, "p95": 3000, "mean": 1900}
    assert stats["stt_ms"]["n"] == 3  # read from the caller rows, where it is stored
