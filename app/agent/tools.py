import json
import re
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.states import State
from app.db.models import Call, Event, Loan, Outcome, utcnow

MAX_VERIFY_ATTEMPTS = 2
PTP_WINDOW_DAYS = 7
DATE_FORMAT = "%d-%m-%Y"  # DD-MM-YYYY, used for every date going in or out


def start_call(session: Session, loan_id: int, channel: str = "text") -> Call:
    call = Call(loan_id=loan_id, channel=channel, final_state=State.VERIFY)
    session.add(call)
    session.commit()
    return call


def log_event(session: Session, call_id: int, tool: str, args: dict, result: dict) -> dict:
    """Write one audit row per tool call, then hand the result back to the caller."""
    session.add(
        Event(
            call_id=call_id,
            tool_name=tool,
            args_json=json.dumps(args),
            result_json=json.dumps(result),
        )
    )
    session.commit()
    return result


def _parse_date(text: str) -> date | None:
    try:
        return datetime.strptime(text.strip(), DATE_FORMAT).date()
    except ValueError:
        return None


def _normalise_name(name: str) -> str:
    return " ".join(name.lower().split())


def _failed_verifications(session: Session, call_id: int) -> int:
    events = session.scalars(
        select(Event).where(Event.call_id == call_id, Event.tool_name == "verify_identity")
    )
    return sum(1 for e in events if not json.loads(e.result_json).get("verified"))


def verify_identity(session: Session, call_id: int, name: str, dob_or_last4: str) -> dict:
    call = session.get(Call, call_id)
    # Raw DOB / phone digits are not written to the audit log, only which kind was given.
    factor = "dob" if _parse_date(dob_or_last4) else "last4"
    args = {"name": name, "factor": factor}

    if call.ended_at:
        return log_event(session, call_id, "verify_identity", args, {"ok": False, "error": "call_ended"})
    if call.verified:
        return log_event(session, call_id, "verify_identity", args, {"ok": True, "verified": True})

    borrower = session.get(Loan, call.loan_id).borrower
    if factor == "dob":
        factor_ok = _parse_date(dob_or_last4) == borrower.dob
    else:
        digits = re.sub(r"\D", "", dob_or_last4)
        factor_ok = len(digits) == 4 and borrower.phone.endswith(digits)
    name_ok = _normalise_name(name) == _normalise_name(borrower.name)

    if name_ok and factor_ok:
        call.verified = True
        return log_event(session, call_id, "verify_identity", args, {"ok": True, "verified": True})

    failed = _failed_verifications(session, call_id) + 1
    attempts_left = MAX_VERIFY_ATTEMPTS - failed
    result = {"ok": False, "verified": False, "attempts_left": max(attempts_left, 0)}
    if attempts_left <= 0:
        call.ended_at = utcnow()
        call.final_state = State.ENDED
        call.outcome = "verify_failed"
        result["call_ended"] = True
    # Deliberately does not say whether the name or the second factor was wrong.
    return log_event(session, call_id, "verify_identity", args, result)


def _require_verified(session: Session, call_id: int, tool: str, args: dict):
    """Return (call, None) if the tool may run, else (call, error_result)."""
    call = session.get(Call, call_id)
    if call.ended_at:
        return call, log_event(session, call_id, tool, args, {"ok": False, "error": "call_ended"})
    if not call.verified:
        return call, log_event(session, call_id, tool, args, {"ok": False, "error": "not_verified"})
    return call, None


def get_loan_summary(session: Session, call_id: int) -> dict:
    call, error = _require_verified(session, call_id, "get_loan_summary", {})
    if error:
        return error
    loan = session.get(Loan, call.loan_id)
    result = {
        "ok": True,
        "emi_amount": loan.emi_amount,
        "due_date": loan.due_date.strftime(DATE_FORMAT),
        "days_past_due": loan.days_past_due,
        "late_fee": loan.late_fee,
        "total_due": loan.emi_amount + loan.late_fee,
    }
    return log_event(session, call_id, "get_loan_summary", {}, result)


def record_promise_to_pay(
    session: Session, call_id: int, date_str: str, amount: int, today: date | None = None
) -> dict:
    args = {"date": date_str, "amount": amount}
    call, error = _require_verified(session, call_id, "record_promise_to_pay", args)
    if error:
        return error

    def reject(reason: str) -> dict:
        return log_event(session, call_id, "record_promise_to_pay", args, {"ok": False, "error": reason})

    ptp_date = _parse_date(date_str)
    if ptp_date is None:
        return reject("invalid_date_format")
    today = today or date.today()
    if not today <= ptp_date <= today + timedelta(days=PTP_WINDOW_DAYS):
        return reject("date_outside_7_days")

    loan = session.get(Loan, call.loan_id)
    total_due = loan.emi_amount + loan.late_fee
    if type(amount) is not int or amount <= 0 or amount > total_due:
        return reject("invalid_amount")

    session.add(Outcome(call_id=call_id, type="PTP", ptp_date=ptp_date, ptp_amount=amount))
    call.outcome = "PTP"
    result = {"ok": True, "ptp_date": ptp_date.strftime(DATE_FORMAT), "ptp_amount": amount}
    return log_event(session, call_id, "record_promise_to_pay", args, result)
