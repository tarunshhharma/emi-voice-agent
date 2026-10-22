from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import Base, Borrower, Loan


@pytest.fixture
def session():
    """In-memory DB with one fixed borrower (Rahul Sharma) and one overdue loan."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        borrower = Borrower(
            name="Rahul Sharma",
            phone="+919876543210",
            dob=date(1990, 5, 14),
            preferred_language="hinglish",
            city="Pune",
        )
        s.add(borrower)
        s.flush()
        s.add(
            Loan(
                borrower_id=borrower.id,
                product="personal",
                principal=100_000,
                emi_amount=3500,
                due_date=date(2026, 10, 12),
                days_past_due=10,
                late_fee=500,
                segment="overdue_1_30",
            )
        )
        s.commit()
        yield s
