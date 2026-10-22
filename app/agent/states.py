from enum import StrEnum


class State(StrEnum):
    VERIFY = "verify"  # prove who the caller is; nothing about the loan may be said yet
    REMIND = "remind"  # verified: fetch the loan details
    RESOLVE = "resolve"  # agree on an outcome with the caller
    DONE = "done"  # an outcome is recorded: say goodbye
    ENDED = "ended"  # call over


# Which tools exist in each state. Tools the plan adds later (payment link, dispute,
# escalate, ...) get added to these sets as they are built.
ALLOWED_TOOLS = {
    State.VERIFY: {"verify_identity"},
    State.REMIND: {"get_loan_summary"},
    State.RESOLVE: {"record_promise_to_pay"},
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
    if tool == "record_promise_to_pay":
        return State.DONE
    return state


def get_state(call) -> State:
    return State.ENDED if call.ended_at else State(call.final_state)
