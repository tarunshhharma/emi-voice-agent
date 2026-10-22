import json

from app.agent.tools import start_call
from app.pipeline.llm import run_tool
from evals.personas import PERSONAS
from evals.simulate import score

VERIFY = json.dumps({"name": "Rahul Sharma", "dob_or_last4": "3210"})
WRONG_PERSON = next(p for p in PERSONAS if p["id"] == "wrong_person")
HARDSHIP = next(p for p in PERSONAS if p["id"] == "hardship")


def test_every_persona_has_the_fields_the_simulator_needs():
    for p in PERSONAS:
        assert {"id", "segment", "knows", "brief", "expected", "must_not_disclose"} <= set(p)
        assert p["knows"] in {"borrower", "name_only", "nothing"}


def test_personas_cover_all_six_outcomes():
    covered = set().union(*(p["expected"] for p in PERSONAS))
    assert {"PTP", "link_sent", "already_paid", "dispute", "callback", "escalated"} <= covered


def test_wrong_person_passes_when_call_ends_without_disclosure(session):
    call = start_call(session, 1)
    run_tool(session, call.id, "end_call", json.dumps({"reason": "wrong_number"}))
    result = score(session, WRONG_PERSON, call)
    assert result["passed"] and result["disclosure_ok"]


def test_wrong_person_fails_if_loan_summary_was_ever_returned(session):
    # Simulates a broken system: the wrong person somehow got verified and heard the summary.
    call = start_call(session, 1)
    run_tool(session, call.id, "verify_identity", VERIFY)
    run_tool(session, call.id, "get_loan_summary", "{}")
    run_tool(session, call.id, "escalate_to_human", json.dumps({"reason": "x"}))
    call.outcome = "wrong_number"
    result = score(session, WRONG_PERSON, call)
    assert result["disclosure_ok"] is False and result["passed"] is False


def test_wrong_outcome_fails(session):
    call = start_call(session, 1)
    run_tool(session, call.id, "end_call", json.dumps({"reason": "not_available"}))
    assert score(session, HARDSHIP, call)["passed"] is False
