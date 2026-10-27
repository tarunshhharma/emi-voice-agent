import json
import re
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.guardrails import IST, calling_hours_ok
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


# STT writes Indian names in Roman script inconsistently ("Kala" for "Kalla", "Sarma" for
# "Sharma"), so names are compared on a rough phonetic key, and a near miss still counts.
# The DOB or last-4 digits must still match exactly: that factor stops impersonation; the
# name only confirms we have the right person. 0.75 was set on real calls: Sarvam heard
# "Chaman Kalla" as "Saman kalam" (0.82); clearly different names score ~0.5-0.6.
NAME_SIMILARITY = 0.75
_SPELLING_VARIANTS = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("sh", "s"), ("ph", "f"), ("w", "v")]


def _name_key(name: str) -> str:
    key = re.sub(r"[^a-z ]", "", name.lower())
    for variant, plain in _SPELLING_VARIANTS:
        key = key.replace(variant, plain)
    key = re.sub(r"(.)\1+", r"\1", key)  # double letters: "kalla" -> "kala"
    return " ".join(key.split())


def _names_match(said: str, stored: str) -> bool:
    a, b = _name_key(said), _name_key(stored)
    return bool(a) and (a == b or SequenceMatcher(None, a, b).ratio() >= NAME_SIMILARITY)


# Digits and months as callers say them, in English and Hindi (Roman script, as STT writes it).
DIGIT_WORDS = {
    "zero": "0", "oh": "0", "shunya": "0", "one": "1", "ek": "1", "two": "2", "do": "2",
    "three": "3", "teen": "3", "four": "4", "chaar": "4", "char": "4", "five": "5", "paanch": "5",
    "panch": "5", "six": "6", "chhah": "6", "chhe": "6", "che": "6", "chah": "6", "seven": "7",
    "saat": "7", "sat": "7", "eight": "8", "aath": "8", "aat": "8", "nine": "9", "nau": "9",
}
MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
MONTH_WORDS = {name.lower(): n for n, name in enumerate(MONTH_NAMES) if name}
MONTH_WORDS.update({name.lower()[:3]: n for n, name in enumerate(MONTH_NAMES) if name})
MONTH_WORDS.update({"janvari": 1, "farvari": 2, "aprail": 4, "mai": 5, "joon": 6, "jun": 6, "julai": 7,
                    "agast": 8, "sitambar": 9, "aktubar": 10, "navambar": 11, "disambar": 12})


def factor_was_said(dob_or_last4: str, caller_text: str) -> bool:
    """Did the caller actually say this DOB or these last 4 digits during the call?
    Seen live: the model called verify_identity with digits nobody had said."""
    tokens = [DIGIT_WORDS.get(t, t) for t in re.findall(r"[a-z]+|\d+", caller_text.lower())]
    dob = _parse_date(dob_or_last4)
    if dob:
        numbers = {int(t) for t in tokens if t.isdigit()}
        month_said = dob.month in numbers or any(MONTH_WORDS.get(t) == dob.month for t in tokens)
        return dob.year in numbers and dob.day in numbers and month_said
    digits = re.sub(r"\D", "", dob_or_last4)
    return len(digits) == 4 and digits in "".join(t for t in tokens if t.isdigit())


def _spoken_detail(dob_or_last4: str) -> str:
    dob = _parse_date(dob_or_last4)
    if dob:
        return f"date of birth {dob.day} {MONTH_NAMES[dob.month]} {dob.year}"
    return "phone ke aakhri chaar digits " + " ".join(re.sub(r"\D", "", dob_or_last4))


def _read_back_done(session: Session, call_id: int) -> bool:
    return session.scalars(
        select(Event).where(Event.call_id == call_id, Event.tool_name == "verify_read_back")
    ).first() is not None


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
    name_ok = _names_match(name, borrower.name)

    if name_ok and factor_ok:
        call.verified = True
        return log_event(session, call_id, "verify_identity", args, {"ok": True, "verified": True})

    # First mismatch in a call: STT may have misheard ("21 June" as "24 June"), so read back
    # what we understood instead of failing. Always both name and detail, so it never shows
    # which part was wrong, and it only repeats the caller's own words. One per call: the
    # cost is one extra guess for an impostor (three instead of two).
    if not _read_back_done(session, call_id):
        logged = log_event(session, call_id, "verify_read_back", args,
                           {"ok": False, "verified": False, "error": "read_back_and_confirm"})
        # The spoken detail goes to the model only, never into the audit log.
        return {**logged, "read_back": f"naam {name}, {_spoken_detail(dob_or_last4)}"}

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


# --- the other outcomes ---


def _record_outcome(session: Session, call: Call, tool: str, args: dict, result: dict, **fields) -> dict:
    session.add(Outcome(call_id=call.id, **fields))
    call.outcome = fields["type"]
    return log_event(session, call.id, tool, args, result)


def _require_text(session: Session, call_id: int, tool: str, args: dict, value: str) -> dict | None:
    if not value or not value.strip():
        return log_event(session, call_id, tool, args, {"ok": False, "error": "missing_details"})
    return None


def send_payment_link(session: Session, call_id: int, channel: str) -> dict:
    args = {"channel": channel}
    call, error = _require_verified(session, call_id, "send_payment_link", args)
    if error:
        return error
    if channel not in {"sms", "whatsapp"}:
        return log_event(session, call_id, "send_payment_link", args, {"ok": False, "error": "invalid_channel"})
    # Mock: nothing is really sent. The link points at a fake payment page.
    result = {"ok": True, "channel": channel, "link": f"https://pay.example.com/emi/{call.loan_id}"}
    return _record_outcome(session, call, "send_payment_link", args, result,
                           type="link_sent", notes=f"channel: {channel}")


def mark_already_paid(session: Session, call_id: int, utr_or_date: str) -> dict:
    args = {"utr_or_date": utr_or_date}
    call, error = _require_verified(session, call_id, "mark_already_paid", args)
    if error or (error := _require_text(session, call_id, "mark_already_paid", args, utr_or_date)):
        return error
    # We don't argue with the caller: flag it so the reconciliation team checks the payment.
    result = {"ok": True, "flagged_for_reconciliation": True}
    return _record_outcome(session, call, "mark_already_paid", args, result,
                           type="already_paid", notes=f"caller says paid: {utr_or_date.strip()}")


def log_dispute(session: Session, call_id: int, reason: str) -> dict:
    args = {"reason": reason}
    call, error = _require_verified(session, call_id, "log_dispute", args)
    if error or (error := _require_text(session, call_id, "log_dispute", args, reason)):
        return error
    result = {"ok": True, "ticket": f"DSP-{call.id:05d}"}
    return _record_outcome(session, call, "log_dispute", args, result,
                           type="dispute", notes=reason.strip())


def schedule_callback(session: Session, call_id: int, when: str, now: datetime | None = None) -> dict:
    """`when` is DD-MM-YYYY HH:MM in India time. Must be in the future, within 7 days, in calling hours."""
    args = {"when": when}
    call, error = _require_verified(session, call_id, "schedule_callback", args)
    if error:
        return error

    def reject(reason: str) -> dict:
        return log_event(session, call_id, "schedule_callback", args, {"ok": False, "error": reason})

    try:
        at = datetime.strptime(when.strip(), f"{DATE_FORMAT} %H:%M").replace(tzinfo=IST)
    except ValueError:
        return reject("invalid_datetime_format")
    now = now or datetime.now(IST)
    if not now < at <= now + timedelta(days=PTP_WINDOW_DAYS):
        return reject("outside_next_7_days")
    if not calling_hours_ok(at):
        return reject("outside_calling_hours")

    result = {"ok": True, "callback_at": at.strftime(f"{DATE_FORMAT} %H:%M")}
    return _record_outcome(session, call, "schedule_callback", args, result,
                           type="callback", notes=f"callback at {result['callback_at']} IST")


def escalate_to_human(session: Session, call_id: int, reason: str) -> dict:
    """Allowed before verification too (e.g. distress), so it must never return loan data."""
    args = {"reason": reason}
    call = session.get(Call, call_id)
    if call.ended_at:
        return log_event(session, call_id, "escalate_to_human", args, {"ok": False, "error": "call_ended"})
    if error := _require_text(session, call_id, "escalate_to_human", args, reason):
        return error
    result = {"ok": True, "escalated": True}
    return _record_outcome(session, call, "escalate_to_human", args, result,
                           type="escalated", notes=reason.strip())


def end_call(session: Session, call_id: int, reason: str) -> dict:
    """For calls that end with no outcome: wrong number, borrower not available, refuses to verify."""
    args = {"reason": reason}
    call = session.get(Call, call_id)
    if call.ended_at:
        return log_event(session, call_id, "end_call", args, {"ok": False, "error": "call_ended"})
    if reason not in {"wrong_number", "not_available", "refused_verification"}:
        return log_event(session, call_id, "end_call", args, {"ok": False, "error": "invalid_reason"})
    call.ended_at = utcnow()
    call.final_state = State.ENDED
    call.outcome = reason
    return log_event(session, call_id, "end_call", args, {"ok": True, "call_ended": True})
