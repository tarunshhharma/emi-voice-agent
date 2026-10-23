"""One call's agent brain for the voice path: the DB session, the history, and run_turn.

Everything here is blocking (OpenAI's sync client, SQLite), so the voice session runs
each method in a worker thread with asyncio.to_thread. The same run_turn as text mode.
"""
import logging
import threading

from sqlalchemy.orm import Session

from app.agent.guardrails import check_call_allowed
from app.agent.prompts import opening
from app.agent.tools import start_call
from app.db.models import Loan, Turn, utcnow
from app.pipeline.llm import run_turn

log = logging.getLogger(__name__)
INTERRUPTED_NOTE = (
    "The caller interrupted your last reply, so they may not have heard all of it. "
    "Don't assume they heard any figures or questions from it; repeat what matters briefly."
)


class CallBlocked(Exception):
    """The call must not happen (do-not-call, outside calling hours)."""


class CallAgent:
    def __init__(self, session: Session, client, loan_id: int, allow_off_hours: bool = False):
        self.session = session
        self.client = client
        loan = session.get(Loan, loan_id)
        if loan is None:
            raise CallBlocked(f"no loan {loan_id}")
        self.borrower = loan.borrower
        reason = check_call_allowed(self.borrower)
        if reason == "outside_calling_hours" and allow_off_hours:
            log.warning("calling-hours check bypassed (ALLOW_OFF_HOURS)")
            reason = None
        if reason:
            raise CallBlocked(reason)
        self.call = start_call(session, loan_id, channel="browser")
        self.first_line = opening(self.borrower.name)
        self.messages = [{"role": "assistant", "content": self.first_line}]
        self._turn_start = len(self.messages)  # where the latest turn's messages begin
        # A thread can't be cancelled: if the caller hangs up mid-turn, the turn finishes in
        # the background. The lock makes close() wait for it instead of pulling the DB away.
        self._lock = threading.Lock()

    def turn(self, text: str, on_text, interrupted: bool = False, on_tool_start=None,
             replaces_previous: bool = False) -> bool:
        """One caller turn. on_text gets each checked sentence as soon as it is ready.
        interrupted: the caller cut off our last reply. on_tool_start: see run_turn.
        replaces_previous: the caller kept talking before Priya said anything, so the last
        turn (their first fragment and her unspoken reply) is replaced by this one, which
        carries both fragments. Tools that turn already ran can't be undone.
        Returns True when the call has ended."""
        with self._lock:  # also waits for an interrupted turn still finishing in its thread
            if replaces_previous:
                del self.messages[self._turn_start:]
            self._turn_start = len(self.messages)
            if interrupted:
                # The history holds the whole last reply, but the caller only heard part of it.
                self.messages.append({"role": "developer", "content": INTERRUPTED_NOTE})
            self.messages.append({"role": "user", "content": text})
            run_turn(self.client, self.session, self.call.id, self.messages, on_text=on_text,
                     on_tool_start=on_tool_start)
            self.session.refresh(self.call)  # tools change the call row (verified, ended)
            return self.call.ended_at is not None

    def save_turn(self, caller_text: str | None, agent_text: str, timings: dict, interrupted: bool) -> None:
        """One row for what the caller said and one for what Priya actually got to say,
        with the latency of each stage. caller_text is None for the opening."""
        with self._lock:
            if caller_text is not None:
                self.session.add(Turn(call_id=self.call.id, speaker="caller", text=caller_text,
                                      stt_ms=timings.get("stt_ms")))
            self.session.add(Turn(
                call_id=self.call.id, speaker="agent", text=agent_text, interrupted=interrupted,
                llm_ttft_ms=timings.get("llm_ttft_ms"), tts_first_byte_ms=timings.get("tts_first_byte_ms"),
                endpoint_ms=timings.get("endpoint_ms"), voice_to_voice_ms=timings.get("voice_to_voice_ms"),
                barge_in_ms=timings.get("barge_in_ms"), filler_ms=timings.get("filler_ms"),
            ))
            self.session.commit()

    def close(self) -> None:
        """The caller hung up (or the call ended): record the end time and release the DB."""
        with self._lock:
            if self.call.ended_at is None:
                self.call.ended_at = utcnow()
                self.session.commit()
            self.session.close()
