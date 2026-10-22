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


def test_duplicate_message_items_in_one_response_are_spoken_once(session):
    # Seen live: the model returned the same question as two message items in one response.
    from types import SimpleNamespace as NS

    question = "Aapko link SMS par bheju ya WhatsApp par, ji?"
    events = []
    for _ in range(2):
        events.append(NS(type="response.output_text.delta", delta=question))
        events.append(NS(type="response.output_text.done"))
    events.append(NS(type="response.completed", response=NS(output=[NS(type="message")] * 2)))

    call = start_call(session, 1)
    heard = []
    said = run_turn(FakeClient([events]), session, call.id, [], on_text=heard.append)
    assert said == question
    assert len(heard) == 1


def test_agent_cannot_hang_up_on_its_own_question(session):
    # Found by evals: the agent asked for the DOB and called end_call in the same turn.
    from types import SimpleNamespace as NS

    item = NS(type="function_call", name="end_call",
              arguments=json.dumps({"reason": "refused_verification"}), call_id="c1")
    asks_then_hangs_up = [
        NS(type="response.output_text.delta", delta="Aapki date of birth kya hai?"),
        NS(type="response.output_text.done"),
        NS(type="response.completed", response=NS(output=[item])),
    ]
    call = start_call(session, 1)
    run_turn(FakeClient([asks_then_hangs_up, text_response("Ji, bataiye.")]), session, call.id, [],
             on_text=lambda d: None)
    assert call.ended_at is None
