import json
import logging
import os
import re
from datetime import date

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.agent import guardrails, tools
from app.agent.prompts import state_instructions, system_prompt
from app.agent.states import ALLOWED_TOOLS, State, get_state, next_state
from app.db.models import Call, Loan, utcnow

load_dotenv()
log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5  # stops a model that keeps calling tools forever
SENTENCE_END = re.compile(r"[.!?।]\s")  # needs the space after, so "3.5" is not split

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "verify_identity",
        "description": "Check the caller's identity against our records. Call this before any loan detail is shared.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Caller's full name"},
                "dob_or_last4": {
                    "type": "string",
                    "description": "Date of birth as DD-MM-YYYY, or the last 4 digits of their phone number",
                },
            },
            "required": ["name", "dob_or_last4"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_loan_summary",
        "description": "Get the EMI amount, due date, days overdue and late fee. Only works after verification.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "record_promise_to_pay",
        "description": "Record the caller's promise to pay. Date must be within the next 7 days; amount cannot exceed the total due.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Payment date as DD-MM-YYYY"},
                "amount": {"type": "integer", "description": "Amount in rupees"},
            },
            "required": ["date", "amount"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _dispatch(session: Session, call_id: int, name: str, arguments: str) -> dict:
    try:
        args = json.loads(arguments)
        if name == "verify_identity":
            return tools.verify_identity(session, call_id, args["name"], args["dob_or_last4"])
        if name == "get_loan_summary":
            return tools.get_loan_summary(session, call_id)
        if name == "record_promise_to_pay":
            return tools.record_promise_to_pay(session, call_id, args["date"], args["amount"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"ok": False, "error": "bad_arguments"}
    return {"ok": False, "error": "unknown_tool"}


def run_tool(session: Session, call_id: int, name: str, arguments: str) -> dict:
    """Run one tool the model asked for, if the call's current state allows it.
    The model only names a tool; our code decides whether it runs."""
    call = session.get(Call, call_id)
    state = get_state(call)
    if name not in ALLOWED_TOOLS[state]:
        return tools.log_event(
            session, call_id, name, {}, {"ok": False, "error": "tool_not_allowed", "state": state}
        )
    result = _dispatch(session, call_id, name, arguments)
    call.final_state = next_state(state, name, result)
    session.commit()
    return result


def run_turn(client, session: Session, call_id: int, messages: list, on_text=print) -> str:
    """Stream one agent turn. Runs any tool calls, feeds results back, and repeats
    until the model answers in plain text. Returns what was said."""
    model = os.getenv("OPENAI_MODEL") or "gpt-6-luna"
    spoken = []
    call = session.get(Call, call_id)
    loan = session.get(Loan, call.loan_id)

    def say(sentence: str) -> None:
        """Nothing reaches the caller without passing the guardrails first."""
        text = sentence.strip()
        if not text:
            return
        result = guardrails.check_reply(text, call.verified, loan)
        if not result.allowed:
            log.warning("blocked reply (%s): %s", result.reason, guardrails.mask_pii(text))
            tools.log_event(
                session, call_id, "guardrail_block", {"reason": result.reason}, {"blocked": text}
            )
        if spoken and spoken[-1] == result.text:
            return  # two blocked sentences in a row would repeat the same safe line
        spoken.append(result.text)
        on_text(result.text + " ")

    for _ in range(MAX_TOOL_ROUNDS):
        # Re-read the state every round: a tool call can move the call to the next state,
        # and the model should only see that state's instructions and tools.
        state = get_state(call)
        request = {
            "model": model,
            "instructions": f"{system_prompt(date.today())}\n\nCurrent step: {state_instructions(state)}",
            "input": messages,
            "reasoning": {"effort": "none"},  # no hidden thinking time: lower time-to-first-token
            "stream": True,
        }
        allowed = [t for t in TOOL_SCHEMAS if t["name"] in ALLOWED_TOOLS[state]]
        if allowed:
            request["tools"] = allowed
        stream = client.responses.create(**request)
        output = []
        buffer = ""
        for event in stream:
            if event.type == "response.output_text.delta":
                buffer += event.delta
                while match := SENTENCE_END.search(buffer):  # speak each finished sentence
                    say(buffer[: match.end()])
                    buffer = buffer[match.end():]
            elif event.type == "response.completed":
                output = event.response.output
        say(buffer)  # whatever is left when the reply ends

        messages.extend(output)  # keep the model's own output in the history
        tool_calls = [item for item in output if item.type == "function_call"]
        if not tool_calls:
            break
        for tool_call in tool_calls:
            result = run_tool(session, call_id, tool_call.name, tool_call.arguments)
            messages.append(
                {"type": "function_call_output", "call_id": tool_call.call_id, "output": json.dumps(result)}
            )
    if get_state(call) == State.DONE:  # the goodbye has been said: close the call
        call.ended_at = utcnow()
        session.commit()
    return " ".join(spoken)
