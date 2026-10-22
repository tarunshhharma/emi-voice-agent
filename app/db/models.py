from datetime import date, datetime, timezone

from sqlalchemy import ForeignKey, Text, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

engine = create_engine("sqlite:///emi.db")
SessionLocal = sessionmaker(engine)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Borrower(Base):
    __tablename__ = "borrowers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    phone: Mapped[str]
    dob: Mapped[date]
    preferred_language: Mapped[str]  # hi / en / hinglish
    city: Mapped[str]
    dnd_flag: Mapped[bool] = mapped_column(default=False)  # "do not call"


class Loan(Base):
    __tablename__ = "loans"

    id: Mapped[int] = mapped_column(primary_key=True)
    borrower_id: Mapped[int] = mapped_column(ForeignKey("borrowers.id"))
    product: Mapped[str]  # two_wheeler / personal / consumer_durable
    principal: Mapped[int]
    emi_amount: Mapped[int]
    due_date: Mapped[date]
    days_past_due: Mapped[int]
    late_fee: Mapped[int]
    segment: Mapped[str]  # seed tag so evals can target each path

    borrower: Mapped[Borrower] = relationship()


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    loan_id: Mapped[int] = mapped_column(ForeignKey("loans.id"))
    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    ended_at: Mapped[datetime | None]
    channel: Mapped[str]  # browser / phone / text
    final_state: Mapped[str | None]
    outcome: Mapped[str | None]
    verified: Mapped[bool] = mapped_column(default=False)


class Turn(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(primary_key=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("calls.id"))
    speaker: Mapped[str]  # agent / caller
    text: Mapped[str] = mapped_column(Text)
    stt_ms: Mapped[int | None]
    llm_ttft_ms: Mapped[int | None]
    tts_first_byte_ms: Mapped[int | None]
    interrupted: Mapped[bool] = mapped_column(default=False)


class Outcome(Base):
    __tablename__ = "outcomes"

    id: Mapped[int] = mapped_column(primary_key=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("calls.id"))
    type: Mapped[str]  # PTP / link_sent / already_paid / dispute / callback / escalated
    ptp_date: Mapped[date | None]
    ptp_amount: Mapped[int | None]
    notes: Mapped[str | None] = mapped_column(Text)


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("calls.id"))
    tool_name: Mapped[str]
    args_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text)
    ts: Mapped[datetime] = mapped_column(default=utcnow)
