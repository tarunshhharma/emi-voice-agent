from enum import StrEnum


class State(StrEnum):
    VERIFY = "verify"  # prove who the caller is; nothing about the loan may be said yet
    REMIND = "remind"  # verified: fetch the loan details
    RESOLVE = "resolve"  # agree on an outcome with the caller
    DONE = "done"  # an outcome is recorded: say goodbye
    ENDED = "ended"  # call over


# Tools that record the call's one outcome. Any of them succeeding moves the call to DONE.
OUTCOME_TOOLS = {
    "record_promise_to_pay",
    "send_payment_link",
    "mark_already_paid",
    "log_dispute",
    "schedule_callback",
    "escalate_to_human",
}

ALLOWED_TOOLS = {
    # escalate is allowed before verification (distress, anger) because it reveals nothing.
    State.VERIFY: {"verify_identity", "escalate_to_human", "end_call"},
    State.REMIND: {"get_loan_summary", "schedule_callback", "escalate_to_human"},
    State.RESOLVE: {
        "record_promise_to_pay",
        "send_payment_link",
        "mark_already_paid",
        "log_dispute",
        "schedule_callback",
        "escalate_to_human",
    },
    State.DONE: set(),
    State.ENDED: set(),
}


def next_state(state: State, tool: str, result: dict) -> State:
    """Code, not the model, decides when the call moves on."""
    if result.get("call_ended") or result.get("error") == "call_ended":
        return State.ENDED
    if not result.get("ok"):
        return state  # a failed tool call never advances the call
    if tool == "verify_identity":
        return State.REMIND
    if tool == "get_loan_summary":
        return State.RESOLVE
    if tool in OUTCOME_TOOLS:
        return State.DONE
    return state


def get_state(call) -> State:
    return State.ENDED if call.ended_at else State(call.final_state)
