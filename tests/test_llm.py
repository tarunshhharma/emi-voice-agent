import json

from app.agent.tools import start_call
from app.pipeline.llm import MAX_TOOL_ROUNDS, run_tool, run_turn
from tests.fakes import FakeClient, text_response, tool_response


def test_run_tool_bad_arguments(session):
    call = start_call(session, 1)
    assert run_tool(session, call.id, "verify_identity", "not json")["error"] == "bad_arguments"
    assert run_tool(session, call.id, "verify_identity", "{}")["error"] == "bad_arguments"


def test_run_tool_unknown_tool(session):
    call = start_call(session, 1)
    # Not in any state's allowed set, so it is refused before it is even dispatched.
    assert run_tool(session, call.id, "waive_fee", "{}")["error"] == "tool_not_allowed"


def test_turn_runs_tool_then_answers(session):
    call = start_call(session, 1)
    client = FakeClient([
        tool_response("verify_identity", {"name": "Rahul Sharma", "dob_or_last4": "3210"}),
        text_response("Shukriya."),
    ])
    messages = [{"role": "user", "content": "main Rahul Sharma, last four 3210"}]
    said = run_turn(client, session, call.id, messages, on_text=lambda d: None)

    assert said == "Shukriya."
    assert call.verified is True
    output = [m for m in messages if isinstance(m, dict) and m.get("type") == "function_call_output"]
    assert json.loads(output[0]["output"])["verified"] is True


def test_turn_stops_after_max_tool_rounds(session):
    call = start_call(session, 1)
    looping = [tool_response("verify_identity", {"name": "x", "dob_or_last4": "0000"})] * 20
    client = FakeClient(looping)
    run_turn(client, session, call.id, [], on_text=lambda d: None)
    assert client.calls <= MAX_TOOL_ROUNDS
