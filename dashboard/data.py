"""Everything the dashboard shows, as plain queries over emi.db. No Streamlit here, so it can be tested.

Text that reaches the screen (transcripts, tool results) goes through mask_pii, the same as logs.
"""
import json

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.guardrails import mask_pii
from app.db.models import Borrower, Call, Event, Loan, Outcome, Turn
from app.pipeline.metrics import summary

SLOW_TURN_MS = 6000  # a voice-to-voice wait this long feels broken: flag the call
NEEDS_FOLLOW_UP = {"dispute", "escalated", "already_paid", "callback"}  # a person has work to do


def calls(session: Session, channels: list[str]) -> pd.DataFrame:
    rows = session.execute(
        select(Call.id, Call.started_at, Call.channel, Call.outcome, Call.verified, Call.ended_at,
               Loan.id.label("loan_id"), Loan.segment, Borrower.name)
        .join(Loan, Call.loan_id == Loan.id).join(Borrower, Loan.borrower_id == Borrower.id)
        .where(Call.channel.in_(channels)).order_by(Call.id.desc())
    ).all()
    return pd.DataFrame(rows, columns=["call", "started", "channel", "outcome", "verified", "ended",
                                       "loan", "segment", "borrower"])


def headline(session: Session, channels: list[str]) -> dict:
    df = calls(session, channels)
    ids = df["call"].tolist()
    ptp = session.scalars(select(Outcome).where(Outcome.call_id.in_(ids), Outcome.type == "PTP")).all()
    stats = summary(session, "browser")  # only voice calls are timed, whatever the filter
    return {
        "calls": len(df),
        "verified_pct": round(100 * df["verified"].mean()) if len(df) else 0,
        "with_outcome_pct": round(100 * df["outcome"].notna().mean()) if len(df) else 0,
        "ptp_count": len(ptp),
        "ptp_amount": sum(o.ptp_amount or 0 for o in ptp),
        "voice_to_voice_p50": stats.get("voice_to_voice_ms", {}).get("p50"),
    }


def outcomes(session: Session, channels: list[str]) -> pd.DataFrame:
    df = calls(session, channels)
    counts = df["outcome"].fillna("no outcome").value_counts()
    return counts.rename_axis("outcome").reset_index(name="calls")


def latency_by_stage(session: Session) -> pd.DataFrame:
    """p50 / p95 per stage over browser calls (only voice calls are timed)."""
    rows = [{"stage": stage.removesuffix("_ms"), **values} for stage, values in summary(session, "browser").items()]
    return pd.DataFrame(rows, columns=["stage", "n", "p50", "p95", "mean"])


def latency_per_turn(session: Session) -> pd.DataFrame:
    """Each timed agent turn in order, for a chart of how latency moves over time."""
    rows = session.execute(
        select(Turn.id, Turn.call_id, Turn.voice_to_voice_ms, Turn.filler_ms, Turn.llm_ttft_ms, Turn.interrupted)
        .join(Call, Turn.call_id == Call.id)
        .where(Call.channel == "browser", Turn.speaker == "agent", Turn.voice_to_voice_ms.is_not(None))
        .order_by(Turn.id)
    ).all()
    return pd.DataFrame(rows, columns=["turn", "call", "voice_to_voice_ms", "filler_ms", "llm_ms", "interrupted"])


def safety(session: Session, channels: list[str]) -> dict:
    """Compliance numbers. Disclosure is checked independently of the output filter: did the
    loan-summary tool ever return data in a call that was never verified?"""
    ids = calls(session, channels)["call"].tolist()
    events = session.scalars(select(Event).where(Event.call_id.in_(ids))).all()
    blocks: dict[str, int] = {}
    for e in events:
        if e.tool_name == "guardrail_block":
            reason = json.loads(e.args_json).get("reason", "unknown")
            blocks[reason] = blocks.get(reason, 0) + 1
    verified = {c.id for c in session.scalars(select(Call).where(Call.id.in_(ids), Call.verified))}
    leaked = {e.call_id for e in events
              if e.tool_name == "get_loan_summary" and json.loads(e.result_json).get("ok") and e.call_id not in verified}
    return {
        "disclosures_before_verification": len(leaked),
        "guardrail_blocks": blocks,
        "failed_verifications": sum(1 for e in events if e.tool_name == "verify_identity"
                                    and not json.loads(e.result_json).get("verified")),
        "deferred_tools": sum(1 for e in events if e.tool_name == "deferred_tool"),
        "refused_tools": sum(1 for e in events if json.loads(e.result_json).get("error") == "tool_not_allowed"),
    }


def barge_ins(session: Session) -> dict:
    rows = session.scalars(
        select(Turn.barge_in_ms).join(Call, Turn.call_id == Call.id)
        .where(Call.channel == "browser", Turn.speaker == "agent", Turn.interrupted)
    ).all()
    timed = sorted(v for v in rows if v is not None)
    return {"count": len(rows), "p50_ms": timed[len(timed) // 2] if timed else None}


def flagged(session: Session, channels: list[str]) -> pd.DataFrame:
    """Calls a person should look at, and why."""
    df = calls(session, channels)
    reasons: dict[int, list[str]] = {int(c): [] for c in df["call"]}
    for call_id, outcome in zip(df["call"], df["outcome"]):
        if outcome in NEEDS_FOLLOW_UP:
            reasons[int(call_id)].append(f"follow-up: {outcome}")
        if outcome == "verify_failed":
            reasons[int(call_id)].append("verification failed")
    for e in session.scalars(select(Event).where(Event.call_id.in_(list(reasons)), Event.tool_name == "guardrail_block")):
        reasons[e.call_id].append("guardrail: " + json.loads(e.args_json).get("reason", "unknown"))
    for t in session.scalars(select(Turn).where(Turn.call_id.in_(list(reasons)), Turn.voice_to_voice_ms >= SLOW_TURN_MS)):
        reasons[t.call_id].append(f"slow turn: {t.voice_to_voice_ms / 1000:.1f} s")
    out = df[df["call"].map(lambda c: bool(reasons[int(c)]))].copy()
    out["why"] = out["call"].map(lambda c: ", ".join(dict.fromkeys(reasons[int(c)])))  # dedupe, keep order
    return out[["call", "started", "borrower", "segment", "outcome", "why"]]


def transcript(session: Session, call_id: int) -> pd.DataFrame:
    rows = session.scalars(select(Turn).where(Turn.call_id == call_id).order_by(Turn.id)).all()
    return pd.DataFrame(
        [{"who": "Priya" if t.speaker == "agent" else "Caller", "said": mask_pii(t.text),
          "voice_to_voice_ms": t.voice_to_voice_ms, "interrupted": t.interrupted} for t in rows],
        columns=["who", "said", "voice_to_voice_ms", "interrupted"])


def tool_events(session: Session, call_id: int) -> pd.DataFrame:
    rows = session.scalars(select(Event).where(Event.call_id == call_id).order_by(Event.id)).all()
    return pd.DataFrame(
        [{"time": e.ts, "tool": e.tool_name, "args": mask_pii(e.args_json), "result": mask_pii(e.result_json)} for e in rows],
        columns=["time", "tool", "args", "result"])
