from collections import Counter

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.models import Borrower, Loan
from app.db.seed import SEGMENT_COUNTS, seed


def test_seed_creates_every_segment():
    bind = create_engine("sqlite://")  # in-memory, so the real emi.db is untouched
    seed(bind)

    with Session(bind) as session:
        assert len(session.scalars(select(Borrower)).all()) == 200
        loans = session.scalars(select(Loan)).all()

    assert Counter(loan.segment for loan in loans) == SEGMENT_COUNTS
    for loan in loans:
        if loan.segment == "pre_due":
            assert loan.days_past_due == 0 and loan.late_fee == 0
        else:
            assert loan.days_past_due > 0 and loan.late_fee > 0
        assert loan.emi_amount > 0 and loan.emi_amount % 50 == 0


from datetime import date as _date
PINNED_FIRST_BORROWER = ("Aryan Maharaj", _date(1972, 7, 22))  # computed on Windows, 24 Oct 2026


def borrowers_of(seed_date):
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session
    from app.db.models import Borrower
    from app.db.seed import seed
    engine = create_engine("sqlite://")
    seed(engine, today=seed_date)
    with Session(engine) as s:
        return [(b.name, b.dob, b.phone, b.preferred_language, b.dnd_flag) for b in s.scalars(select(Borrower).order_by(Borrower.id))]


def test_borrowers_are_the_same_on_any_day():
    # Found 24 Oct: DOBs counted back from today, so they moved every day.
    from datetime import date
    assert borrowers_of(date(2026, 10, 24)) == borrowers_of(date(2026, 12, 1))


def test_first_borrower_is_pinned_so_every_platform_must_agree():
    # Found 24 Oct: the Linux container seeded different borrowers than Windows. This exact
    # value fails on any machine whose seed drifts (CI runs on Linux).
    from datetime import date
    first = borrowers_of(date(2026, 10, 24))[0]
    assert first[:2] == PINNED_FIRST_BORROWER
