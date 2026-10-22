from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.agent.guardrails import (
    IST,
    POLITE_LINE,
    VERIFY_LINE,
    calling_hours_ok,
    check_call_allowed,
    check_reply,
    mask_pii,
)
from app.agent.tools import start_call
from app.db.models import Event, Loan
from app.pipeline.llm import run_turn
from tests.fakes import FakeClient, text_response


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, 22, hour, minute, tzinfo=IST)


# --- calling hours and do-not-call ---


@pytest.mark.parametrize(
    "when, allowed",
    [(at(7, 59), False), (at(8, 0), True), (at(12), True), (at(18, 59), True), (at(19, 0), False), (at(23), False)],
)
def test_calling_hours(when, allowed):
    assert calling_hours_ok(when) is allowed


def test_calling_hours_use_india_time_not_the_server_clock():
    # 03:00 UTC is 08:30 in India (allowed); 14:00 UTC is 19:30 in India (blocked).
    assert calling_hours_ok(datetime(2026, 10, 22, 3, 0, tzinfo=timezone.utc)) is True
    assert calling_hours_ok(datetime(2026, 10, 22, 14, 0, tzinfo=timezone.utc)) is False


def test_check_call_allowed():
    ok = SimpleNamespace(dnd_flag=False)
    dnd = SimpleNamespace(dnd_flag=True)
    assert check_call_allowed(ok, at(10)) is None
    assert check_call_allowed(ok, at(22)) == "outside_calling_hours"
    assert check_call_allowed(dnd, at(10)) == "do_not_call"


# --- no loan details before verification ---


@pytest.fixture
def loan(session):
    return session.get(Loan, 1)  # EMI 3500, late fee 500, total 4000, principal 100000, due 12-10-2026


@pytest.mark.parametrize(
    "reply",
    [
        "Aapki EMI teen hazaar paanch sau rupaye hai.",  # amount as words
        "Your payment of Rs 3500 is pending.",
        "You owe 4,000 rupees.",
        "Aapka loan 100000 ka hai.",
        "Payment 12-10-2026 ko due thi.",
        "Aapki due date twelve October thi.",
        "Late fee lag chuki hai.",
        "Aapka loan overdue hai.",
    ],
)
def test_loan_details_are_blocked_before_verification(reply, loan):
    result = check_reply(reply, verified=False, loan=loan)
    assert result.allowed is False
    assert result.reason == "loan_detail_before_verification"
    assert result.text == VERIFY_LINE


@pytest.mark.parametrize(
    "reply",
    [
        "Namaste, main Priya bol rahi hoon ABC Finance ki taraf se.",
        "Kripya apna poora naam aur date of birth batayein.",
        "Phone number ke aakhri 4 digits batayein.",
        "Could you please confirm your full name?",
    ],
)
def test_verification_questions_are_allowed_before_verification(reply, loan):
    assert check_reply(reply, verified=False, loan=loan).allowed is True


def test_same_details_are_allowed_after_verification(loan):
    reply = "Aapki EMI teen hazaar paanch sau rupaye hai, due date 12-10-2026 thi."
    assert check_reply(reply, verified=True, loan=loan).allowed is True


# --- threats, shaming, third parties (blocked even when verified) ---


@pytest.mark.parametrize(
    "reply, reason",
    [
        ("Hum aapke khilaf legal action lenge.", "threat"),
        ("Warna police aapko giraftaar kar legi.", "threat"),
        ("Hum kanooni karwai karenge.", "threat"),
        ("We will take you to court.", "threat"),
        ("Aapko sharam nahi aati?", "shaming"),
        ("Aap toh defaulter hain.", "shaming"),
        ("Hum aapke parivaar ko bata denge.", "third_party_contact"),
        ("Hum aapke office mein call karenge.", "third_party_contact"),
        ("We will tell your family.", "third_party_contact"),
        ("Your employer will know.", "third_party_contact"),
    ],
)
def test_threats_shaming_and_third_party_contact_are_blocked(reply, reason, loan):
    result = check_reply(reply, verified=True, loan=loan)
    assert (result.allowed, result.reason, result.text) == (False, reason, POLITE_LINE)


@pytest.mark.parametrize(
    "reply",
    [
        "Fir se bataiye, aap kab payment karenge?",  # Hindi "fir" = "again"; must not look like FIR
        "Koi baat nahi, aap apni suvidha se bataiye.",
        "Dhanyavaad, aapka din shubh ho.",
    ],
)
def test_polite_replies_pass_when_verified(reply, loan):
    assert check_reply(reply, verified=True, loan=loan).allowed is True


# --- logging ---


def test_mask_pii_hides_phone_numbers_and_dates():
    masked = mask_pii("Call +919876543210, DOB 14-05-1990, also 9876543210")
    assert "9876543210" not in masked and "1990" not in masked
    assert masked.count("[phone]") == 2 and "[date]" in masked


# --- wired into the reply stream ---


def stream_of(*deltas):
    """A fake model reply that arrives in several small pieces, like a real stream."""
    from types import SimpleNamespace as NS

    events = [NS(type="response.output_text.delta", delta=d) for d in deltas]
    events.append(NS(type="response.completed", response=NS(output=[NS(type="message")])))
    return events


def test_leak_in_a_reply_is_replaced_and_logged(session):
    call = start_call(session, 1)
    client = FakeClient([text_response("Aapki EMI teen hazaar paanch sau rupaye hai.")])
    heard = []
    said = run_turn(client, session, call.id, [], on_text=heard.append)

    assert said == VERIFY_LINE
    assert "hazaar" not in "".join(heard)
    block = [e for e in session.scalars(select(Event)) if e.tool_name == "guardrail_block"]
    assert len(block) == 1 and "loan_detail_before_verification" in block[0].args_json


def test_only_the_bad_sentence_is_replaced(session):
    call = start_call(session, 1)
    client = FakeClient([stream_of("Namaste ji. ", "Aapka loan ", "overdue hai. ", "Dhanyavaad.")])
    heard = []
    run_turn(client, session, call.id, [], on_text=heard.append)

    assert heard[0].strip() == "Namaste ji."
    assert heard[1].strip() == VERIFY_LINE
    assert heard[2].strip() == "Dhanyavaad."


def test_repeated_safe_lines_are_not_spoken_twice(session):
    call = start_call(session, 1)
    client = FakeClient([stream_of("Aapka loan hai. ", "Aapki EMI baaki hai.")])
    heard = []
    run_turn(client, session, call.id, [], on_text=heard.append)
    assert [h.strip() for h in heard] == [VERIFY_LINE]
