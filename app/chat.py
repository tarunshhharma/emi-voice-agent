"""Terminal chat with the agent (text mode). Usage: python -m app.chat [loan_id] [--any-time]"""
import logging
import sys

from openai import OpenAI
from sqlalchemy import select

from app.agent.guardrails import check_call_allowed
from app.agent.prompts import opening
from app.agent.tools import start_call
from app.db.models import Loan, Outcome, SessionLocal
from app.pipeline.llm import run_turn


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Priya writes Devanagari; the Windows console default can't print it
    logging.basicConfig(level=logging.WARNING, format="\n[log] %(message)s")
    ids = [a for a in sys.argv[1:] if not a.startswith("--")]
    loan_id = int(ids[0]) if ids else 1
    any_time = "--any-time" in sys.argv
    client = OpenAI()  # reads OPENAI_API_KEY from the environment

    with SessionLocal() as session:
        loan = session.get(Loan, loan_id)
        borrower = loan.borrower
        # Synthetic data only: shown so you know what to type when playing the borrower.
        print(f"[dev] loan {loan_id} ({loan.segment}): {borrower.name}, "
              f"DOB {borrower.dob.strftime('%d-%m-%Y')}, phone ends {borrower.phone[-4:]}")

        reason = check_call_allowed(borrower)
        if reason == "outside_calling_hours" and any_time:
            print("[dev] WARNING: calling-hours check bypassed with --any-time")
            reason = None
        if reason:
            print(f"[dev] call blocked: {reason}")
            return

        call = start_call(session, loan_id, channel="text")
        first_line = opening(borrower.name, borrower.preferred_language)
        messages = [{"role": "assistant", "content": first_line}]
        print(f"\nPriya: {first_line}")

        while True:
            try:
                text = input("\nYou: ").strip()
            except EOFError:
                break
            if text.lower() in {"", "quit", "exit"}:
                break
            messages.append({"role": "user", "content": text})
            print("\nPriya: ", end="", flush=True)
            run_turn(client, session, call.id, messages, on_text=lambda d: print(d, end="", flush=True))
            print()
            session.refresh(call)  # tools change the call row (verified, ended)
            if call.ended_at:
                break

        outcomes = session.scalars(select(Outcome).where(Outcome.call_id == call.id)).all()
        print(f"\n[dev] call {call.id}: verified={call.verified}, outcome={call.outcome}, "
              f"ended={call.ended_at is not None}")
        for o in outcomes:
            print(f"[dev] outcome row: {o.type} {o.ptp_date} Rs {o.ptp_amount}")


if __name__ == "__main__":
    main()
