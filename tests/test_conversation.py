"""The voice path's agent loop: a blocking agent turn in a thread, its sentences spoken as they arrive."""
import asyncio
import threading

import pytest

from app.pipeline.call import CallAgent, CallBlocked
from app.pipeline.metrics import TurnTimer
from app.pipeline.session import FILLER_LINE, VoiceSession, from_list
from tests.fakes import FakeClient, text_response, tool_response


class FakeTTS:
    def __init__(self):
        self.heard = []

    async def stream(self, sentences):
        async for s in sentences:
            self.heard.append(s)
            yield b"pcm"


class FakeAgent:
    """Blocking, like CallAgent: says its lines from a worker thread, one by one."""

    first_line = "Namaste, kya main Rahul ji se baat kar rahi hoon?"

    def __init__(self, replies, end_after=None, fail_on=None):
        self.replies = replies  # caller text -> sentences to say
        self.end_after = end_after
        self.fail_on = fail_on
        self.threads = set()
        self.interrupted_flags = []
        self.saved = []  # (caller_text, agent_text, timings, interrupted) per turn

    def save_turn(self, caller_text, agent_text, timings, interrupted):
        self.saved.append((caller_text, agent_text, timings, interrupted))

    def mark_last_reply_interrupted(self, barge_in_ms):
        self.marked = barge_in_ms

    def turn(self, text, on_text, interrupted=False, on_tool_start=None, replaces_previous=False):
        self.threads.add(threading.current_thread().name)
        self.interrupted_flags.append(interrupted)
        if text == self.fail_on:
            raise RuntimeError("openai down")
        for sentence in self.replies[text]:
            on_text(sentence + " ")
        return text == self.end_after


def make_session(agent):
    sent = []

    async def send_json(m):
        sent.append(m)

    async def send_audio(b):
        pass

    tts = FakeTTS()
    return VoiceSession(None, None, send_json, tts=tts, send_audio=send_audio, agent=agent), sent, tts


def test_turn_runs_in_a_worker_thread_and_its_sentences_are_spoken_in_order():
    agent = FakeAgent({"haan": ["Dhanyavaad.", "Apni DOB bataiye?"]})
    session, sent, tts = make_session(agent)
    ended = asyncio.run(session.respond("haan"))
    assert ended is False
    assert threading.main_thread().name not in agent.threads  # the event loop never blocked on it
    assert [s.strip() for s in tts.heard] == ["Dhanyavaad.", "Apni DOB bataiye?"]


def test_converse_speaks_opening_then_each_turn_until_the_call_ends():
    agent = FakeAgent({"haan": ["Dhanyavaad."], "bye": ["Shukriya, namaste."]}, end_after="bye")
    session, sent, tts = make_session(agent)

    async def run():
        session.user_turns.put_nowait(("haan", TurnTimer()))
        session.user_turns.put_nowait(("bye", TurnTimer()))
        await asyncio.wait_for(session.converse(), 5)

    asyncio.run(run())
    spoken = [s.strip() for s in tts.heard if s != FILLER_LINE]  # the filler is pre-synthesized too
    assert spoken == [agent.first_line, "Dhanyavaad.", "Shukriya, namaste."]
    assert sent[-1] == {"type": "ended"}


def test_a_failed_turn_is_reported_and_the_call_continues():
    agent = FakeAgent({"haan": ["Dhanyavaad."], "bye": ["Namaste."]}, end_after="bye", fail_on="oops")
    session, sent, tts = make_session(agent)

    async def run():
        for text in ["oops", "haan", "bye"]:
            session.user_turns.put_nowait((text, TurnTimer()))
        await asyncio.wait_for(session.converse(), 5)

    asyncio.run(run())
    errors = [m for m in sent if m["type"] == "error"]
    assert len(errors) == 1 and "openai down" in errors[0]["message"]
    assert sent[-1] == {"type": "ended"}


def test_call_agent_opens_a_call_and_runs_a_real_turn(session):
    client = FakeClient([text_response("Kripya apni date of birth bataiye. ")])
    agent = CallAgent(session, client, loan_id=1, allow_off_hours=True)
    assert "Rahul Sharma" in agent.first_line and agent.call.channel == "browser"
    said = []
    assert agent.turn("haan main Rahul bol raha hoon", said.append) is False
    assert said == ["Kripya apni date of birth bataiye. "]
    assert agent.messages[1] == {"role": "user", "content": "haan main Rahul bol raha hoon"}


def test_call_agent_verify_tool_moves_the_call_on(session):
    client = FakeClient([
        tool_response("verify_identity", {"name": "Rahul Sharma", "dob_or_last4": "14-05-1990"}),
        text_response("Dhanyavaad Rahul ji. "),
    ])
    agent = CallAgent(session, client, loan_id=1, allow_off_hours=True)
    agent.turn("Rahul Sharma, 14 May 1990", lambda s: None)
    assert agent.call.verified is True


def test_do_not_call_borrower_is_blocked(session):
    from app.db.models import Loan
    session.get(Loan, 1).borrower.dnd_flag = True
    session.commit()
    with pytest.raises(CallBlocked, match="do_not_call"):
        CallAgent(session, FakeClient([]), loan_id=1, allow_off_hours=True)


def test_close_marks_a_hung_up_call_as_ended(session):
    agent = CallAgent(session, FakeClient([]), loan_id=1, allow_off_hours=True)
    call_id = agent.call.id
    agent.close()
    from app.db.models import Call
    assert session.get(Call, call_id).ended_at is not None


# --- barge-in ---

import time

from app.pipeline.session import BARGE_IN_FRAMES
from tests.test_session import FakeSTT, FakeVAD

FRAME = bytes(640)


class SlowAgent(FakeAgent):
    """Says one sentence, then keeps "thinking" until released: a reply the caller can interrupt."""

    def __init__(self):
        super().__init__({})
        self.release = threading.Event()
        self.started = threading.Event()

    def turn(self, text, on_text, interrupted=False, on_tool_start=None, replaces_previous=False):
        self.interrupted_flags.append(interrupted)
        if text == "ruko":
            return False
        on_text("Aapka EMI teen hazaar paanch sau rupaye hai, ")
        self.started.set()
        self.release.wait(5)
        on_text("aur late fee paanch sau.")
        return False


def voice_session(agent, speech_frames):
    sent = []

    async def send_json(m):
        sent.append(m)

    async def send_audio(b):
        pass

    session = VoiceSession(FakeVAD(speech_frames), FakeSTT(), send_json, tts=FakeTTS(),
                           send_audio=send_audio, agent=agent)
    return session, sent


def test_caller_talking_over_a_reply_cancels_it_and_the_next_turn_knows():
    agent = SlowAgent()
    session, sent = voice_session(agent, range(0, 2 * BARGE_IN_FRAMES))

    async def run():
        conversation = asyncio.create_task(session.converse())
        session.user_turns.put_nowait(("haan", TurnTimer()))
        while not agent.started.is_set():
            await asyncio.sleep(0.01)
        # The fake opening is 3 bytes long, so it "finishes playing" almost at once; on Windows
        # time.monotonic() ticks every ~15 ms, so wait past that tick before the caller speaks.
        await asyncio.sleep(0.05)
        for _ in range(2 * BARGE_IN_FRAMES):  # the caller speaks over Priya
            await session.on_frame(FRAME)
        first_reply = session.reply_task
        assert sent[-1]["type"] in ("pause", "vad") and not first_reply.cancelled()  # paused, not yet decided
        await session.on_final("ruko", TurnTimer())  # not a backchannel: a real interruption
        await asyncio.sleep(0.05)
        assert first_reply.cancelled()
        agent.release.set()  # the LLM thread finishes on its own, unheard
        await asyncio.sleep(0.2)  # "ruko" was queued by on_final and becomes the next turn
        conversation.cancel()

    asyncio.run(run())
    assert any(m["type"] == "pause" for m in sent)
    assert [m for m in sent if m["type"] == "barge_in"] == [{"type": "barge_in", "continuation": False}]
    assert agent.interrupted_flags == [False, True]
    spoken = [m["text"] for m in sent if m["type"] == "agent_text"]
    assert "aur late fee paanch sau." not in spoken  # finished after the barge-in: never spoken


def test_a_short_sound_over_priya_is_not_a_barge_in():
    session, sent = voice_session(FakeAgent({}), range(0, BARGE_IN_FRAMES - 1))
    session.playing_until = time.monotonic() + 10  # her audio is still playing

    async def run():
        for _ in range(BARGE_IN_FRAMES + 5):
            await session.on_frame(FRAME)

    asyncio.run(run())
    assert not any(m["type"] == "barge_in" for m in sent)


def test_the_opening_cannot_be_interrupted():
    session, sent = voice_session(FakeAgent({}), range(0, 2 * BARGE_IN_FRAMES))
    session.playing_until = time.monotonic() + 10
    session.no_barge_in_until = time.monotonic() + 10  # what converse() sets for the opening

    async def run():
        for _ in range(2 * BARGE_IN_FRAMES):
            await session.on_frame(FRAME)

    asyncio.run(run())
    assert not any(m["type"] == "barge_in" for m in sent)


def test_audio_still_playing_in_the_browser_counts_as_busy():
    session, sent = voice_session(FakeAgent({}), range(0))

    class OneSecondTTS:
        async def stream(self, sentences):
            async for _ in sentences:
                yield bytes(32_000)  # 1 s of 16 kHz 16-bit audio, sent at once

    session.tts = OneSecondTTS()
    asyncio.run(session.speak(from_list(["Namaste."])))
    assert session.agent_busy()  # sent instantly, but the caller hears it for a second
    assert session.playing_until - time.monotonic() > 0.9


# --- filler while a tool runs ---

from app.pipeline import session as session_module


class ToolAgent(FakeAgent):
    """Starts a tool call (on_tool_start) before saying its reply, like a verify turn."""

    def __init__(self, tool="verify_identity", speak_first=False):
        super().__init__({})
        self.tool, self.speak_first = tool, speak_first

    def turn(self, text, on_text, interrupted=False, on_tool_start=None, replaces_previous=False):
        if self.speak_first:
            on_text("Ek minute. ")
        on_tool_start(self.tool)
        time.sleep(0.1)  # the tool round trip
        on_text("Dhanyavaad, aap verify ho gaye. ")
        return False


def run_one_turn(agent):
    session_module._filler_pcm = b"\x00\x00" * 1600  # pretend the filler was prepared (0.1 s)
    session, sent = voice_session(agent, range(0))
    session.timer = TurnTimer(caller_stopped=time.monotonic())
    asyncio.run(session.respond("Rahul Sharma, 14 May 1990"))
    texts = [m for m in sent if m["type"] == "agent_text"]
    return session, texts


def test_filler_plays_when_a_tool_starts_before_anything_is_said():
    session, texts = run_one_turn(ToolAgent())
    assert texts[0]["text"] == session_module.FILLER_LINE and texts[0].get("filler")
    assert texts[1]["text"] == "Dhanyavaad, aap verify ho gaye."
    d = session.timer.durations()
    assert d["filler_ms"] is not None and d["filler_ms"] < d["voice_to_voice_ms"]
    assert session.said[0] == session_module.FILLER_LINE  # saved as part of what the caller heard


def test_no_filler_before_hanging_up():
    session, texts = run_one_turn(ToolAgent(tool="end_call"))
    assert all(not m.get("filler") for m in texts)

# "No filler after Priya already spoke" is guaranteed in run_turn (deterministic, tested in
# test_llm.py); the session's own check is a second line that a test here could only hit by race.


# --- pause, then decide: backchannels and continuations ---

from app.pipeline.session import ANSWER_WINDOW_S, is_backchannel


def test_backchannel_words():
    assert is_backchannel("Haan.") and is_backchannel("haan haan, theek hai") and is_backchannel("हाँ")
    assert not is_backchannel("ruko") and not is_backchannel("haan par amount galat hai")
    assert not is_backchannel("")


def held_session(remaining_s):
    session, sent = voice_session(FakeAgent({}), range(0, 2 * BARGE_IN_FRAMES))
    session.playing_until = time.monotonic() + remaining_s  # Priya's audio still playing

    async def speak_over():
        for _ in range(2 * BARGE_IN_FRAMES):
            await session.on_frame(FRAME)
    asyncio.run(speak_over())
    assert session.held and any(m["type"] == "pause" for m in sent)
    return session, sent


def test_haan_while_priya_talks_resumes_her_and_is_not_a_turn():
    session, sent = held_session(remaining_s=5)
    asyncio.run(session.on_final("Haan.", TurnTimer()))
    assert sent[-1] == {"type": "resume"} and not session.held
    assert session.user_turns.empty()
    assert not any(m["type"] == "barge_in" for m in sent)


def test_haan_in_her_last_second_is_an_answer():
    session, sent = held_session(remaining_s=ANSWER_WINDOW_S / 2)
    asyncio.run(session.on_final("Haan ji.", TurnTimer()))
    assert sent[-1] == {"type": "resume"}
    assert session.user_turns.get_nowait()[0] == "Haan ji."


def test_noise_with_no_words_resumes_her():
    session, sent = held_session(remaining_s=5)
    asyncio.run(session.on_final("", TurnTimer()))
    assert sent[-1] == {"type": "resume"} and session.user_turns.empty()


def test_a_real_interruption_after_the_pause_is_a_barge_in():
    session, sent = held_session(remaining_s=5)
    asyncio.run(session.on_final("Ruko, amount galat hai.", TurnTimer()))
    assert any(m["type"] == "barge_in" for m in sent) and session.interrupted
    assert session.user_turns.get_nowait()[0] == "Ruko, amount galat hai."


def test_pause_times_out_if_no_transcript_comes(monkeypatch):
    monkeypatch.setattr(session_module, "HOLD_TIMEOUT_S", 0.05)
    session, sent = voice_session(FakeAgent({}), range(0, 2 * BARGE_IN_FRAMES))  # speech, then silence
    session.playing_until = time.monotonic() + 5

    async def run():
        for _ in range(2 * BARGE_IN_FRAMES + 30):  # long enough for the VAD to end the utterance
            await session.on_frame(FRAME)
        await asyncio.sleep(0.2)

    asyncio.run(run())
    assert sent[-1] == {"type": "resume"}


class ThinkingAgent(FakeAgent):
    """Thinks (says nothing) until released: the caller can continue before Priya speaks."""

    def __init__(self):
        super().__init__({})
        self.calls = []  # (text, replaces_previous)
        self.release = threading.Event()
        self.started = threading.Event()

    def turn(self, text, on_text, interrupted=False, on_tool_start=None, replaces_previous=False):
        self.calls.append((text, replaces_previous))
        if len(self.calls) == 1:
            self.started.set()
            self.release.wait(5)
            on_text("Ji, aapki DOB kya hai? ")  # too late: never spoken
            return False
        on_text("Dhanyavaad. ")
        return False


def test_caller_continuing_before_priya_speaks_is_one_merged_turn():
    agent = ThinkingAgent()
    session, sent = voice_session(agent, range(0, 2 * BARGE_IN_FRAMES))

    async def run():
        conversation = asyncio.create_task(session.converse())
        session.user_turns.put_nowait(("Meri date of birth hai", TurnTimer()))
        while not agent.started.is_set():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        for _ in range(2 * BARGE_IN_FRAMES):  # the caller carries on after a pause
            await session.on_frame(FRAME)
        await session.on_final("chaudah May unnees sau nabbe", TurnTimer())
        agent.release.set()
        await asyncio.sleep(0.3)
        conversation.cancel()

    asyncio.run(run())
    assert [m for m in sent if m["type"] == "barge_in"] == [{"type": "barge_in", "continuation": True}]
    assert agent.calls[1] == ("Meri date of birth hai chaudah May unnees sau nabbe", True)
    assert [row[0] for row in agent.saved] == [None, "Meri date of birth hai chaudah May unnees sau nabbe"]
    assert not session.interrupted  # nothing real was said, so nothing was "missed"


def test_call_agent_replaces_the_rolled_back_fragment(session):
    client = FakeClient([text_response("Ji, aapki DOB? "), text_response("Dhanyavaad. ")])
    agent = CallAgent(session, client, loan_id=1, allow_off_hours=True)
    agent.turn("Meri DOB hai", lambda s: None)
    agent.turn("Meri DOB hai chaudah May", lambda s: None, replaces_previous=True)
    users = [m["content"] for m in agent.messages if isinstance(m, dict) and m.get("role") == "user"]
    assert users == ["Meri DOB hai chaudah May"]  # the fragment and its unspoken reply are gone


def test_reply_that_arrived_while_paused_was_never_heard_so_it_is_a_continuation():
    # Seen live: Priya's answer to the first fragment was generated and sent while she was
    # paused for the caller; it never played, so the caller was still mid-sentence.
    session, sent = held_session(remaining_s=2)  # busy: her (unheard) audio is queued
    session.reply_text = "Meri date of birth hai"
    session.timer.first_audio = session.paused_at + 0.5  # her audio started only after the pause
    asyncio.run(session.on_final("chaudah May unnees sau nabbe", TurnTimer()))
    assert [m for m in sent if m["type"] == "barge_in"] == [{"type": "barge_in", "continuation": True}]
    assert session.continuation_text == "Meri date of birth hai" and not session.interrupted


def test_barge_in_during_the_playback_tail_marks_the_saved_reply():
    # Seen in real calls: TTS finished sending (and the turn was saved) before the caller cut in,
    # so the barge-in worked but never reached the turns table.
    session, sent = held_session(remaining_s=5)
    session.reply_text = "haan"  # a reply exists; its task is long done
    session.timer.first_audio = session.paused_at - 1  # and it was playing before the pause

    async def run():
        await session.on_final("Ruko, amount dobara batao.", TurnTimer())
        await asyncio.sleep(0.1)  # the row update runs in a thread
    asyncio.run(run())
    assert session.agent.marked == session.hold_ms and session.interrupted


def test_call_agent_marks_its_latest_reply_as_interrupted(session):
    from app.db.models import Turn
    agent = CallAgent(session, FakeClient([]), loan_id=1, allow_off_hours=True)
    agent.save_turn(None, "Namaste...", {}, interrupted=False)
    agent.save_turn("haan", "Aapki EMI...", {"voice_to_voice_ms": 3000}, interrupted=False)
    agent.mark_last_reply_interrupted(320)
    rows = session.query(Turn).filter_by(call_id=agent.call.id, speaker="agent").order_by(Turn.id).all()
    assert [(r.interrupted, r.barge_in_ms) for r in rows] == [(False, None), (True, 320)]
