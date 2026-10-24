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
NEEDS_CALLER_ANSWER = {"end_call", "verify_identity"}  # never in the same turn as our own question
SENTENCE_END = re.compile(r"[.!?।]\s")  # needs the space after, so "3.5" is not split

def _schema(tool_name: str, description: str, /, **properties) -> dict:
    """Strict function schema: every property is required and nothing extra is allowed.
    The `/` makes the first two positional-only, so a property can itself be called "name"."""
    return {
        "type": "function",
        "name": tool_name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


def _text(description: str) -> dict:
    return {"type": "string", "description": description}


TOOL_SCHEMAS = [
    _schema(
        "verify_identity",
        "Check the caller's identity against our records. Call this before any loan detail is shared.",
        name=_text("Caller's full name, exactly as the caller said it"),
        dob_or_last4=_text("Date of birth as DD-MM-YYYY, or the last 4 digits of their phone number"),
    ),
    _schema(
        "get_loan_summary",
        "Get the EMI amount, due date, days overdue and late fee. Only works after verification.",
    ),
    _schema(
        "record_promise_to_pay",
        "Record the caller's promise to pay. Date must be within the next 7 days; amount cannot exceed the total due.",
        date=_text("Payment date as DD-MM-YYYY"),
        amount={"type": "integer", "description": "Amount in rupees"},
    ),
    _schema(
        "send_payment_link",
        "Send the caller a payment link so they can pay now.",
        channel={"type": "string", "enum": ["sms", "whatsapp"]},
    ),
    _schema(
        "mark_already_paid",
        "The caller says they already paid. Flag it for reconciliation; do not argue.",
        utr_or_date=_text("Payment reference (UTR) or the date they paid, as the caller said it"),
    ),
    _schema(
        "log_dispute",
        "The caller disputes the amount or the loan itself. Opens a ticket; stop asking for payment.",
        reason=_text("Short summary of the dispute in the caller's words"),
    ),
    _schema(
        "schedule_callback",
        "Book a callback when the caller can't talk now. Must be within 7 days, between 8 am and 7 pm.",
        when=_text("Callback time as DD-MM-YYYY HH:MM (24-hour, India time)"),
    ),
    _schema(
        "escalate_to_human",
        "Hand the call to a human agent: hardship (job loss, illness), anger, legal threats, "
        "asks not to be called, or repeated confusion.",
        reason=_text("Why a human is needed"),
    ),
    _schema(
        "end_call",
        "End the call when it can't continue: wrong number, borrower not available, or caller refuses to verify.",
        reason={"type": "string", "enum": ["wrong_number", "not_available", "refused_verification"]},
    ),
]

# How each tool's JSON arguments map onto the Python function in tools.py.
TOOL_FUNCTIONS = {
    "verify_identity": lambda s, c, a: tools.verify_identity(s, c, a["name"], a["dob_or_last4"]),
    "get_loan_summary": lambda s, c, a: tools.get_loan_summary(s, c),
    "record_promise_to_pay": lambda s, c, a: tools.record_promise_to_pay(s, c, a["date"], a["amount"]),
    "send_payment_link": lambda s, c, a: tools.send_payment_link(s, c, a["channel"]),
    "mark_already_paid": lambda s, c, a: tools.mark_already_paid(s, c, a["utr_or_date"]),
    "log_dispute": lambda s, c, a: tools.log_dispute(s, c, a["reason"]),
    "schedule_callback": lambda s, c, a: tools.schedule_callback(s, c, a["when"]),
    "escalate_to_human": lambda s, c, a: tools.escalate_to_human(s, c, a["reason"]),
    "end_call": lambda s, c, a: tools.end_call(s, c, a["reason"]),
}


def _dispatch(session: Session, call_id: int, name: str, arguments: str) -> dict:
    if name not in TOOL_FUNCTIONS:
        return {"ok": False, "error": "unknown_tool"}
    try:
        return TOOL_FUNCTIONS[name](session, call_id, json.loads(arguments))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return {"ok": False, "error": "bad_arguments"}


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


def run_turn(client, session: Session, call_id: int, messages: list, on_text=print, on_tool_start=None) -> str:
    """Stream one agent turn. Runs any tool calls, feeds results back, and repeats
    until the model answers in plain text. Returns what was said.
    on_tool_start(name): called once, when the model starts its first tool call before
    saying anything, so the voice path can fill the silence of the tool round trips."""
    model = os.getenv("OPENAI_MODEL") or "gpt-6-luna"
    spoken = []
    tool_announced = False
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
        if not result.text:
            return  # not speech at all (e.g. tool-call JSON): drop it rather than say a stock line
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
        spoken_item = None  # output_index of the one message item we speak in this response
        for event in stream:
            if event.type == "response.output_item.added" and event.item.type == "function_call":
                # Each tool round trip costs a full extra LLM request (~1-1.5 s). The stream
                # announces the call long before it completes, so the caller can hear a filler now.
                if on_tool_start and not spoken and not tool_announced:
                    tool_announced = True
                    on_tool_start(event.item.name)
            elif event.type == "response.output_item.added" and event.item.type == "message":
                # One response can hold a "commentary" and then a "final_answer" message that
                # repeat each other, often reworded, so the caller would hear it twice. Speak
                # the first (it arrives first: lower latency); later ones stay in the history only.
                if spoken_item is None:
                    spoken_item = event.output_index
                else:
                    log.info("not speaking extra message item (phase=%s)", getattr(event.item, "phase", None))
            elif getattr(event, "output_index", spoken_item) != spoken_item:
                continue  # text of an extra message item
            elif event.type == "response.output_text.delta":
                buffer += event.delta
                while match := SENTENCE_END.search(buffer):  # speak each finished sentence
                    say(buffer[: match.end()])
                    buffer = buffer[match.end():]
            elif event.type == "response.output_text.done":
                say(buffer)
                buffer = ""
            elif event.type == "response.completed":
                output = event.response.output
        say(buffer)  # whatever is left when the reply ends

        messages.extend(output)  # keep the model's own output in the history
        tool_calls = [item for item in output if item.type == "function_call"]
        if not tool_calls:
            break
        deferred = False
        for tool_call in tool_calls:
            if tool_call.name in NEEDS_CALLER_ANSWER and any(s.endswith("?") for s in spoken):
                deferred = True
                # Found by evals (end_call) and the voice path (verify_identity): the model asked
                # for the DOB, then hung up or "verified" with invented digits in the same turn.
                # It can't have the answer to a question it just asked, so wait for it.
                # Logged under its own name so it never counts as a failed verification.
                result = tools.log_event(
                    session, call_id, "deferred_tool", {"tool": tool_call.name},
                    {"ok": False, "error": "wait_for_caller_answer"},
                )
            else:
                result = run_tool(session, call_id, tool_call.name, tool_call.arguments)
            messages.append(
                {"type": "function_call_output", "call_id": tool_call.call_id, "output": json.dumps(result)}
            )
        if deferred:
            # The question is already asked; the turn is over until the caller answers. Another
            # request here only made the model apologise and repeat itself (seen live), and cost ~1-2 s.
            break
    if get_state(call) == State.DONE:  # the goodbye has been said: close the call
        call.ended_at = utcnow()
        session.commit()
    return " ".join(spoken)
