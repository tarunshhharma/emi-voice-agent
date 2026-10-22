from datetime import date

import pytest
from sqlalchemy import select

from app.agent.tools import (
    get_loan_summary,
    record_promise_to_pay,
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


def test_wrong_factor_fails_without_ending_call(session, call):
    result = verify_identity(session, call.id, "Rahul Sharma", "1111")
    assert result == {"ok": False, "verified": False, "attempts_left": 1}
    assert call.verified is False and call.ended_at is None


def test_wrong_name_fails(session, call):
    assert verify_identity(session, call.id, "Amit Verma", "3210")["verified"] is False


def test_two_failures_end_the_call(session, call):
    verify_identity(session, call.id, "Rahul Sharma", "1111")
    result = verify_identity(session, call.id, "Rahul Sharma", "2222")
    assert result["call_ended"] is True
    assert call.ended_at is not None and call.outcome == "verify_failed"
    # Even the correct answer is refused once the call has ended.
    assert verify_identity(session, call.id, "Rahul Sharma", "3210")["error"] == "call_ended"


def test_failure_does_not_leak_which_part_was_wrong(session, call):
    result = verify_identity(session, call.id, "Rahul Sharma", "1111")
    assert set(result) == {"ok", "verified", "attempts_left"}


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
