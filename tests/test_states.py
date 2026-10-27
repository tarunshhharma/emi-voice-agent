import json
from datetime import date, timedelta

from sqlalchemy import select

from app.agent.states import ALLOWED_TOOLS, OUTCOME_TOOLS, State, get_state, next_state
from app.agent.tools import start_call
from app.db.models import Event, Outcome
from app.pipeline.llm import run_tool, run_turn
from tests.fakes import FakeClient, text_response, tool_response

VERIFY_ARGS = json.dumps({"name": "Rahul Sharma", "dob_or_last4": "3210"})


def ptp_args(amount: int = 3500) -> str:
    day = (date.today() + timedelta(days=2)).strftime("%d-%m-%Y")
    return json.dumps({"date": day, "amount": amount})


def test_nothing_that_touches_the_loan_is_allowed_before_verification():
    assert ALLOWED_TOOLS[State.VERIFY] == {"verify_identity", "escalate_to_human", "end_call"}


def test_every_outcome_tool_moves_the_call_to_done():
    for tool in OUTCOME_TOOLS:
        assert next_state(State.RESOLVE, tool, {"ok": True}) == State.DONE


def test_no_tools_once_an_outcome_is_recorded():
    assert ALLOWED_TOOLS[State.DONE] == set() and ALLOWED_TOOLS[State.ENDED] == set()


def test_call_starts_in_verify(session):
    assert get_state(start_call(session, 1)) == State.VERIFY


def test_next_state_moves_only_on_success():
    assert next_state(State.VERIFY, "verify_identity", {"ok": True}) == State.REMIND
    assert next_state(State.VERIFY, "verify_identity", {"ok": False}) == State.VERIFY
    assert next_state(State.REMIND, "get_loan_summary", {"ok": True}) == State.RESOLVE
    assert next_state(State.RESOLVE, "record_promise_to_pay", {"ok": True}) == State.DONE
    failed_twice = {"ok": False, "call_ended": True}
    assert next_state(State.VERIFY, "verify_identity", failed_twice) == State.ENDED


def test_tool_from_a_later_state_is_refused(session):
    call = start_call(session, 1)
    result = run_tool(session, call.id, "record_promise_to_pay", ptp_args())
    assert result["error"] == "tool_not_allowed"
    assert session.scalars(select(Outcome)).all() == []
    assert get_state(call) == State.VERIFY
    # The refusal is audited like any other tool call.
    event = session.scalars(select(Event)).one()
    assert json.loads(event.result_json)["error"] == "tool_not_allowed"


def test_loan_summary_is_refused_before_verification_even_if_model_asks(session):
    call = start_call(session, 1)
    assert run_tool(session, call.id, "get_loan_summary", "{}")["error"] == "tool_not_allowed"


def test_full_flow_walks_the_states(session):
    call = start_call(session, 1)
    assert run_tool(session, call.id, "verify_identity", VERIFY_ARGS)["verified"] is True
    assert get_state(call) == State.REMIND
    assert run_tool(session, call.id, "get_loan_summary", "{}")["ok"] is True
    assert get_state(call) == State.RESOLVE


def test_second_outcome_is_refused_once_one_is_recorded(session):
    call = start_call(session, 1)
    run_tool(session, call.id, "verify_identity", VERIFY_ARGS)
    run_tool(session, call.id, "get_loan_summary", "{}")
    assert run_tool(session, call.id, "record_promise_to_pay", ptp_args())["ok"] is True
    assert get_state(call) == State.DONE
    assert run_tool(session, call.id, "record_promise_to_pay", ptp_args())["error"] == "tool_not_allowed"
    assert len(session.scalars(select(Outcome)).all()) == 1


def test_two_failed_verifications_end_the_call(session):
    call = start_call(session, 1)
    bad = json.dumps({"name": "Rahul Sharma", "dob_or_last4": "0000"})
    run_tool(session, call.id, "verify_identity", bad)  # first mismatch: read back
    run_tool(session, call.id, "verify_identity", bad)
    run_tool(session, call.id, "verify_identity", bad)
    assert get_state(call) == State.ENDED


def test_model_only_sees_tools_for_the_current_state(session):
    call = start_call(session, 1)
    client = FakeClient([
        tool_response("verify_identity", {"name": "Rahul Sharma", "dob_or_last4": "3210"}),
        tool_response("get_loan_summary", {}),
        text_response("Aapki EMI saadhe teen hazaar hai."),
    ])
    said = [{"role": "user", "content": "Rahul Sharma, last four 3 2 1 0"}]  # verify needs the caller's digits
    run_turn(client, session, call.id, said, on_text=lambda d: None)

    offered = [{t["name"] for t in r["tools"]} for r in client.requests]
    expected = [ALLOWED_TOOLS[State.VERIFY], ALLOWED_TOOLS[State.REMIND], ALLOWED_TOOLS[State.RESOLVE]]
    assert offered == expected
    assert "not verified yet" in client.requests[0]["instructions"]
    assert "Call get_loan_summary now" in client.requests[1]["instructions"]


def test_call_closes_after_the_goodbye(session):
    call = start_call(session, 1)
    client = FakeClient([
        tool_response("verify_identity", {"name": "Rahul Sharma", "dob_or_last4": "3210"}),
        tool_response("get_loan_summary", {}),
        tool_response("record_promise_to_pay", json.loads(ptp_args())),
        text_response("Dhanyavaad, namaste."),
    ])
    said = [{"role": "user", "content": "Rahul Sharma, last four 3 2 1 0"}]
    run_turn(client, session, call.id, said, on_text=lambda d: None)

    assert call.ended_at is not None
    assert "tools" not in client.requests[-1]  # nothing left to call once an outcome exists
