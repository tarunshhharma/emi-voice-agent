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


def test_reworded_final_answer_after_commentary_is_not_spoken(session):
    # Seen live in the voice path: a "commentary" message, then a reworded "final_answer"
    # in the same response. The caller heard the question twice.
    from types import SimpleNamespace as NS

    first = "Kya aap apna poora naam aur date of birth bata sakte hain?"
    second = "Aap apna poora naam aur date of birth bata sakte hain?"
    events = []
    for index, (phase, text) in enumerate([("commentary", first), ("final_answer", second)]):
        events.append(NS(type="response.output_item.added", output_index=index,
                         item=NS(type="message", phase=phase)))
        events.append(NS(type="response.output_text.delta", output_index=index, delta=text))
        events.append(NS(type="response.output_text.done", output_index=index))
    events.append(NS(type="response.completed", response=NS(output=[NS(type="message")] * 2)))

    call = start_call(session, 1)
    heard = []
    said = run_turn(FakeClient([events]), session, call.id, [], on_text=heard.append)
    assert said == first
    assert heard == [first + " "]


def test_verify_in_the_same_turn_as_our_question_is_deferred_and_not_a_failed_attempt(session):
    # Seen live: Priya asked for the DOB, then called verify_identity with invented digits.
    from app.agent.tools import _failed_verifications
    from app.db.models import Event
    from types import SimpleNamespace as NS

    question = text_response("Apni date of birth bataiye?")
    verify = tool_response("verify_identity", {"name": "Rahul Sharma", "dob_or_last4": "1234"})
    asked_then_verified = question[:-1] + [NS(type="response.completed", response=NS(
        output=question[-1].response.output + verify[-1].response.output))]
    client = FakeClient([asked_then_verified, text_response("Main aapke jawab ka intezaar karti hoon. ")])

    call = start_call(session, 1)
    run_turn(client, session, call.id, [], on_text=lambda d: None)
    assert _failed_verifications(session, call.id) == 0
    deferred = session.query(Event).filter_by(call_id=call.id, tool_name="deferred_tool").one()
    assert json.loads(deferred.args_json) == {"tool": "verify_identity"}
    assert call.verified is False
    assert client.calls == 1  # the turn ends after deferring: no extra request, no repeated question


def _tool_call_with_announcement(name, arguments, said_first=None):
    """A response that (optionally) says a sentence, then starts a tool call, like the live stream."""
    from types import SimpleNamespace as NS
    item = NS(type="function_call", name=name, arguments=json.dumps(arguments), call_id="call_1")
    events = []
    if said_first:
        events += [NS(type="response.output_item.added", output_index=0, item=NS(type="message", phase="commentary")),
                   NS(type="response.output_text.delta", output_index=0, delta=said_first),
                   NS(type="response.output_text.done", output_index=0)]
    index = 1 if said_first else 0
    events.append(NS(type="response.output_item.added", output_index=index, item=item))
    output = ([NS(type="message")] if said_first else []) + [item]
    events.append(NS(type="response.completed", response=NS(output=output)))
    return events


def test_tool_start_is_announced_once_before_anything_is_said(session):
    client = FakeClient([
        _tool_call_with_announcement("verify_identity", {"name": "Rahul Sharma", "dob_or_last4": "14-05-1990"}),
        _tool_call_with_announcement("get_loan_summary", {}),
        text_response("Dhanyavaad Rahul ji. "),
    ])
    call = start_call(session, 1)
    announced = []
    run_turn(client, session, call.id, [], on_text=lambda d: None, on_tool_start=announced.append)
    assert announced == ["verify_identity"]  # the second tool in the same turn is not announced again


def test_no_tool_announcement_after_the_agent_already_spoke(session):
    client = FakeClient([
        _tool_call_with_announcement("verify_identity", {"name": "Rahul Sharma", "dob_or_last4": "14-05-1990"},
                                     said_first="Ek minute, main verify karti hoon. "),
        text_response("Dhanyavaad. "),
    ])
    call = start_call(session, 1)
    announced = []
    run_turn(client, session, call.id, [], on_text=lambda d: None, on_tool_start=announced.append)
    assert announced == []  # she already said something; a filler on top would double up
