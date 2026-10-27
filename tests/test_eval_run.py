import json
from types import SimpleNamespace

from app.agent.tools import log_event, start_call
from app.db.models import Turn
from evals.eval_run import jobs, summarize
from evals.judge import call_record, judge
from evals.personas import PERSONAS


def test_every_run_calls_a_different_borrower_and_peers_never_share_one():
    todo = jobs(PERSONAS, runs=4)
    assert len(todo) == len(PERSONAS) * 4 == 40
    picked = {}
    for persona, nth in todo:
        key = (persona["segment"], persona.get("language"), nth)  # a pool is segment + language
        assert key not in picked, f"{persona['id']} and {picked.get(key)} share a borrower"
        picked[key] = persona["id"]


def row(**over):
    base = {"persona": "p", "outcome_ok": True, "disclosure_ok": True, "no_disclosure": True, "compliance_ok": True,
            "passed": True, "guardrail_blocks": 0, "turns": 6,
            "judge": {"quality": 4, "outcome_fits_caller": "pass", "disclosed_before_verification": False}}
    base.update(over)
    return base


def test_summary_counts_violations_and_disagreements():
    rows = [row(), row(passed=False, compliance_ok=False), row(passed=False, outcome_ok=False),
            row(passed=False, no_disclosure=False), {"persona": "p", "error": "boom"}]
    s = summarize(rows)
    assert s["calls"] == 5 and s["errors"] == 1
    assert s["task_success_pct"] == 75 and s["passed_pct"] == 25
    assert s["compliance_violations"] == 1 and s["disclosure_violations"] == 1
    assert s["judge_outcome_disagreements"] == 1  # code says wrong outcome, judge said it fits


def test_judge_sees_transcript_tool_log_and_outcome(session):
    call = start_call(session, 1, channel="sim")
    session.add_all([Turn(call_id=call.id, speaker="agent", text="Namaste."),
                     Turn(call_id=call.id, speaker="caller", text="Haan ji.")])
    log_event(session, call.id, "verify_identity", {"factor": "dob"}, {"ok": True, "verified": True})
    session.commit()
    record = call_record(session, call.id)
    assert "Priya: Namaste." in record and "Caller: Haan ji." in record
    assert 'verify_identity -> {"ok": true, "verified": true}' in record
    assert "RECORDED OUTCOME: None" in record


def test_judge_parses_the_structured_verdict(session):
    verdict = {"compliance": "pass", "disclosed_before_verification": False, "outcome_fits_caller": "pass",
               "quality": 4, "reason": "no issues"}
    sent = {}

    class Client:
        class responses:
            @staticmethod
            def create(**kw):
                sent.update(kw)
                return SimpleNamespace(output_text=json.dumps(verdict))

    call = start_call(session, 1, channel="sim")
    assert judge(Client, session, call.id) == verdict
    assert sent["text"]["format"]["strict"] is True  # the schema is enforced, not just suggested


def test_tool_calls_are_interleaved_with_the_speech_in_time_order(session):
    # Found on the first eval run: with the tool log after the transcript, the judge thought the
    # loan details came before verification.
    from datetime import datetime, timedelta, timezone
    from app.db.models import Event
    t0 = datetime(2026, 10, 24, 10, 0, tzinfo=timezone.utc)
    call = start_call(session, 1, channel="sim")
    session.add_all([
        Turn(call_id=call.id, speaker="caller", text="DOB 14-05-1990.", ts=t0),
        Event(call_id=call.id, tool_name="verify_identity", args_json="{}", result_json='{"verified": true}',
              ts=t0 + timedelta(seconds=1)),
        Turn(call_id=call.id, speaker="agent", text="Aapki EMI 3500 hai.", ts=t0 + timedelta(seconds=2)),
    ])
    session.commit()
    lines = call_record(session, call.id).splitlines()
    caller, tool, agent = (next(i for i, l in enumerate(lines) if key in l)
                           for key in ("Caller: DOB", "[tool] verify_identity", "Priya: Aapki EMI"))
    assert caller < tool < agent


def test_english_persona_draws_only_english_preference_borrowers(session):
    from app.db.models import Borrower, Loan
    from evals.simulate import pick_loan
    english = session.get(Loan, 1).borrower
    english.preferred_language = "en"
    session.commit()
    assert pick_loan(session, "overdue_1_30", 0, language="en").borrower is english
    import pytest
    with pytest.raises(IndexError):  # there is only one English borrower in the fixture
        pick_loan(session, "overdue_1_30", 1, language="en")
