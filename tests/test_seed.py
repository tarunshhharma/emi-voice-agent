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
