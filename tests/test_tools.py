from datetime import date, datetime

import pytest
from sqlalchemy import select

from app.agent.guardrails import IST
from app.agent.tools import (
    end_call,
    escalate_to_human,
    get_loan_summary,
    log_dispute,
    mark_already_paid,
    record_promise_to_pay,
    schedule_callback,
    send_payment_link,
    start_call,
    verify_identity,
)
from app.db.models import Event, Outcome

TODAY = date(2026, 10, 22)


@pytest.fixture
def call(session):
    return start_call(session, loan_id=1)


@pytest.fixture
def verified_call(session, call):
    verify_identity(session, call.id, "Rahul Sharma", "14-05-1990")
    return call


# --- verify_identity ---


def test_verify_with_dob(session, call):
    result = verify_identity(session, call.id, "Rahul Sharma", "14-05-1990")
    assert result["verified"] is True
    assert call.verified is True


def test_verify_with_last4(session, call):
    assert verify_identity(session, call.id, "Rahul Sharma", "3210")["verified"] is True


def test_name_ignores_case_and_extra_spaces(session, call):
    assert verify_identity(session, call.id, "  rahul   SHARMA ", "3210")["verified"] is True


def test_first_mismatch_is_read_back_not_failed(session, call):
    # STT can mishear a digit; the first mismatch reads the details back instead of failing.
    result = verify_identity(session, call.id, "Rahul Sharma", "1111")
    assert result["error"] == "read_back_and_confirm" and result["verified"] is False
    assert result["read_back"] == "naam Rahul Sharma, phone ke aakhri chaar digits 1 1 1 1"
    from app.agent.tools import _failed_verifications
    assert _failed_verifications(session, call.id) == 0 and call.ended_at is None


def test_read_back_speaks_a_dob_in_words():
    from app.agent.tools import _spoken_detail
    assert _spoken_detail("21-06-1982") == "date of birth 21 June 1982"


def test_a_corrected_answer_after_the_read_back_verifies(session, call):
    verify_identity(session, call.id, "Rahul Sharma", "1111")  # misheard: read back
    assert verify_identity(session, call.id, "Rahul Sharma", "3210")["verified"] is True


def test_wrong_factor_fails_without_ending_call(session, call):
    verify_identity(session, call.id, "Rahul Sharma", "1111")  # read back
    result = verify_identity(session, call.id, "Rahul Sharma", "1111")  # caller confirmed it
    assert result == {"ok": False, "verified": False, "attempts_left": 1}
    assert call.verified is False and call.ended_at is None


def test_wrong_name_fails(session, call):
    assert verify_identity(session, call.id, "Amit Verma", "3210")["verified"] is False


def test_two_failures_end_the_call(session, call):
    verify_identity(session, call.id, "Rahul Sharma", "1111")  # read back (one per call)
    verify_identity(session, call.id, "Rahul Sharma", "1111")  # failure 1
    result = verify_identity(session, call.id, "Rahul Sharma", "2222")  # failure 2
    assert result["call_ended"] is True
    assert call.ended_at is not None and call.outcome == "verify_failed"
    # Even the correct answer is refused once the call has ended.
    assert verify_identity(session, call.id, "Rahul Sharma", "3210")["error"] == "call_ended"


def test_failure_does_not_leak_which_part_was_wrong(session, call):
    # The read-back always repeats both name and detail, whichever was wrong...
    wrong_factor = verify_identity(session, call.id, "Rahul Sharma", "1111")
    assert wrong_factor["read_back"].startswith("naam Rahul Sharma, phone")
    # ...and a real failure says nothing about which part failed.
    result = verify_identity(session, call.id, "Rahul Sharma", "1111")
    assert set(result) == {"ok", "verified", "attempts_left"}


def test_wrong_name_is_read_back_the_same_way(session, call):
    result = verify_identity(session, call.id, "Amit Verma", "3210")
    assert result["read_back"] == "naam Amit Verma, phone ke aakhri chaar digits 3 2 1 0"


def test_read_back_never_stores_the_spoken_digits(session, call):
    from app.db.models import Event
    result = verify_identity(session, call.id, "Rahul Sharma", "01-01-1990")  # wrong DOB: read back
    assert result["read_back"] == "naam Rahul Sharma, date of birth 1 January 1990"  # the model hears it...
    for e in session.query(Event).filter_by(call_id=call.id):  # ...the audit log never stores it
        assert "01-01-1990" not in e.result_json and "1 January 1990" not in e.result_json


def test_factor_must_have_been_said_by_the_caller():
    from app.agent.tools import factor_was_said
    assert factor_was_said("21-06-1982", "Mera naam Aryan Maharaj, date of birth 21 June 1982.")
    assert factor_was_said("14-10-1969", "DOB chaudah... I mean 14 October 1969")
    assert factor_was_said("21-06-1982", "21st joon 1982")  # Hindi month spelling, ordinal
    assert factor_was_said("5092", "last four digits paanch zero nau do")  # digits as words
    assert factor_was_said("5092", "5 0 9 2")
    assert not factor_was_said("21-06-1982", "Haan ji, main Aryan Maharaj bol raha hoon.")
    assert not factor_was_said("21-06-1982", "date of birth 24 June 1982")
    assert not factor_was_said("1234", "Mera naam Rahul Sharma")


def test_audit_log_does_not_store_raw_dob_or_digits(session, call):
    verify_identity(session, call.id, "Rahul Sharma", "14-05-1990")
    event = session.scalars(select(Event)).one()
    assert "1990" not in event.args_json
    assert '"factor": "dob"' in event.args_json


# --- get_loan_summary ---


def test_summary_refused_before_verification(session, call):
    result = get_loan_summary(session, call.id)
    assert result == {"ok": False, "error": "not_verified"}
    assert "emi_amount" not in result


def test_summary_after_verification(session, verified_call):
    result = get_loan_summary(session, verified_call.id)
    assert result["emi_amount"] == 3500
    assert result["late_fee"] == 500
    assert result["total_due"] == 4000
    assert result["due_date"] == "12-10-2026"
    assert result["days_past_due"] == 10


# --- record_promise_to_pay ---


def test_ptp_refused_before_verification(session, call):
    result = record_promise_to_pay(session, call.id, "24-10-2026", 3500, today=TODAY)
    assert result["error"] == "not_verified"
    assert session.scalars(select(Outcome)).all() == []


def test_ptp_recorded(session, verified_call):
    result = record_promise_to_pay(session, verified_call.id, "25-10-2026", 3500, today=TODAY)
    assert result == {"ok": True, "ptp_date": "25-10-2026", "ptp_amount": 3500}
    outcome = session.scalars(select(Outcome)).one()
    assert (outcome.type, outcome.ptp_date, outcome.ptp_amount) == ("PTP", date(2026, 10, 25), 3500)
    assert verified_call.outcome == "PTP"


def test_ptp_allows_today_and_day_seven(session, verified_call):
    assert record_promise_to_pay(session, verified_call.id, "22-10-2026", 100, today=TODAY)["ok"]
    assert record_promise_to_pay(session, verified_call.id, "29-10-2026", 100, today=TODAY)["ok"]


def test_ptp_date_too_far(session, verified_call):
    result = record_promise_to_pay(session, verified_call.id, "30-10-2026", 3500, today=TODAY)
    assert result["error"] == "date_outside_7_days"


def test_ptp_date_in_past(session, verified_call):
    result = record_promise_to_pay(session, verified_call.id, "21-10-2026", 3500, today=TODAY)
    assert result["error"] == "date_outside_7_days"


def test_ptp_bad_date_format(session, verified_call):
    result = record_promise_to_pay(session, verified_call.id, "2026-10-25", 3500, today=TODAY)
    assert result["error"] == "invalid_date_format"


@pytest.mark.parametrize("amount", [0, -100, 4001, "3500"])
def test_ptp_invalid_amount(session, verified_call, amount):
    result = record_promise_to_pay(session, verified_call.id, "25-10-2026", amount, today=TODAY)
    assert result["error"] == "invalid_amount"
    assert session.scalars(select(Outcome)).all() == []


def test_ptp_full_due_amount_is_allowed(session, verified_call):
    assert record_promise_to_pay(session, verified_call.id, "25-10-2026", 4000, today=TODAY)["ok"]


# --- the other outcomes ---

NOW = datetime(2026, 10, 22, 11, 0, tzinfo=IST)  # a Thursday morning in India


def only_outcome(session):
    return session.scalars(select(Outcome)).one()


@pytest.mark.parametrize(
    "tool, args",
    [
        (send_payment_link, ("sms",)),
        (mark_already_paid, ("UTR123456",)),
        (log_dispute, ("amount is wrong",)),
        (schedule_callback, ("23-10-2026 10:00",)),
    ],
)
def test_outcome_tools_refuse_before_verification(session, call, tool, args):
    assert tool(session, call.id, *args)["error"] == "not_verified"
    assert session.scalars(select(Outcome)).all() == []


def test_send_payment_link(session, verified_call):
    result = send_payment_link(session, verified_call.id, "whatsapp")
    assert result["ok"] and result["link"].startswith("https://pay.example.com/")
    assert only_outcome(session).type == "link_sent"
    assert verified_call.outcome == "link_sent"


def test_send_payment_link_rejects_unknown_channel(session, verified_call):
    assert send_payment_link(session, verified_call.id, "email")["error"] == "invalid_channel"


def test_mark_already_paid_flags_for_reconciliation(session, verified_call):
    result = mark_already_paid(session, verified_call.id, "UTR998877")
    assert result == {"ok": True, "flagged_for_reconciliation": True}
    assert "UTR998877" in only_outcome(session).notes


def test_already_paid_needs_some_detail(session, verified_call):
    assert mark_already_paid(session, verified_call.id, "  ")["error"] == "missing_details"


def test_log_dispute_opens_a_ticket(session, verified_call):
    result = log_dispute(session, verified_call.id, "I never took this loan")
    assert result["ok"] and result["ticket"].startswith("DSP-")
    assert only_outcome(session).type == "dispute"


def test_schedule_callback_in_hours(session, verified_call):
    result = schedule_callback(session, verified_call.id, "23-10-2026 10:30", now=NOW)
    assert result == {"ok": True, "callback_at": "23-10-2026 10:30"}
    assert only_outcome(session).type == "callback"


@pytest.mark.parametrize(
    "when, error",
    [
        ("23-10-2026 21:00", "outside_calling_hours"),
        ("23-10-2026 07:30", "outside_calling_hours"),
        ("22-10-2026 10:00", "outside_next_7_days"),  # earlier today: already past
        ("05-11-2026 10:00", "outside_next_7_days"),
        ("tomorrow 10am", "invalid_datetime_format"),
    ],
)
def test_schedule_callback_rejections(session, verified_call, when, error):
    assert schedule_callback(session, verified_call.id, when, now=NOW)["error"] == error
    assert session.scalars(select(Outcome)).all() == []


def test_escalate_works_before_verification_and_reveals_nothing(session, call):
    result = escalate_to_human(session, call.id, "caller lost job, very upset")
    assert result == {"ok": True, "escalated": True}
    assert only_outcome(session).type == "escalated"


def test_end_call_for_wrong_number(session, call):
    assert end_call(session, call.id, "wrong_number") == {"ok": True, "call_ended": True}
    assert call.ended_at is not None and call.outcome == "wrong_number"
    assert session.scalars(select(Outcome)).all() == []  # not one of the six outcomes


def test_end_call_rejects_unknown_reason(session, call):
    assert end_call(session, call.id, "bored")["error"] == "invalid_reason"
    assert call.ended_at is None


def test_stt_spelling_variants_of_the_name_still_verify(session, call):
    # Seen live: Sarvam wrote "Chaman Kala" for "Chaman Kalla", and the right DOB failed.
    for said in ["Rahul Sarma", "Raahul Sharmaa", "rahul  sharma"]:
        assert verify_identity(session, call.id, said, "3210")["verified"] is True, said
        call.verified = False


def test_a_close_name_alone_never_verifies(session, call):
    # The tolerant name match only works with the exact second factor.
    assert verify_identity(session, call.id, "Rahul Sarma", "1111")["verified"] is False
    assert verify_identity(session, call.id, "Amit Verma", "3210")["verified"] is False


def test_name_threshold_boundary_from_real_calls():
    # Set on real calls: STT manglings of the right name pass, clearly different names don't.
    from app.agent.tools import _names_match
    assert _names_match("Saman kalam", "Chaman Kalla")      # Sarvam, real call
    assert _names_match("R.N. Maharaj", "Aryan Maharaj")    # Sarvam, real call
    assert not _names_match("Suresh Kalla", "Chaman Kalla")
    assert not _names_match("Chaman Singh", "Chaman Kalla")
    assert not _names_match("Amit Verma", "Rahul Sharma")
