"""The voice path's agent loop: a blocking agent turn in a thread, its sentences spoken as they arrive."""
import asyncio
import threading

import pytest

from app.pipeline.call import CallAgent, CallBlocked
from app.pipeline.metrics import TurnTimer
from app.pipeline.session import VoiceSession, from_list
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

    def turn(self, text, on_text, interrupted=False):
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
    assert [s.strip() for s in tts.heard] == [agent.first_line, "Dhanyavaad.", "Shukriya, namaste."]
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

    def turn(self, text, on_text, interrupted=False):
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
        await asyncio.sleep(0.05)
        assert session.reply_task.cancelled()
        agent.release.set()  # the LLM thread finishes on its own, unheard
        session.user_turns.put_nowait(("ruko", TurnTimer()))  # what the caller said becomes the next turn
        await asyncio.sleep(0.2)
        conversation.cancel()

    asyncio.run(run())
    assert [m for m in sent if m["type"] == "barge_in"] == [{"type": "barge_in"}]
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
