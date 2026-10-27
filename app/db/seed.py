import logging
import random
from datetime import date, timedelta

from faker import Faker
from sqlalchemy.orm import Session

from app.db.models import Base, Borrower, Loan, engine

log = logging.getLogger(__name__)

# 200 borrowers in total, split so every demo/eval path has data.
SEGMENT_COUNTS = {
    "pre_due": 40,  # EMI due in 3 days
    "overdue_1_30": 50,
    "overdue_31_60": 40,
    "paid_unreconciled": 25,  # caller already paid, our books don't show it yet
    "wrong_number": 20,  # the person who answers is not the borrower
    "hardship": 25,  # job loss etc.
}
DPD_RANGE = {
    "pre_due": (0, 0),
    "overdue_1_30": (1, 30),
    "overdue_31_60": (31, 60),
    "paid_unreconciled": (1, 15),
    "wrong_number": (1, 30),
    "hardship": (5, 45),
}
PRODUCTS = {  # product: (min principal, max principal)
    "two_wheeler": (40_000, 150_000),
    "personal": (50_000, 500_000),
    "consumer_durable": (10_000, 80_000),
}


def make_loan(segment: str, borrower_id: int, today: date) -> Loan:
    product = random.choice(list(PRODUCTS))
    low, high = PRODUCTS[product]
    principal = random.randrange(low, high + 1, 1000)
    tenure_months = random.choice([6, 12, 18, 24])
    # EMI rounded to 50 so it is easy to say aloud ("teen hazaar paanch sau")
    emi_amount = round(principal * 1.12 / tenure_months / 50) * 50

    dpd = random.randint(*DPD_RANGE[segment])
    due_date = today - timedelta(days=dpd) if dpd else today + timedelta(days=3)
    late_fee = 250 + 25 * dpd if dpd else 0

    return Loan(
        borrower_id=borrower_id,
        product=product,
        principal=principal,
        emi_amount=emi_amount,
        due_date=due_date,
        days_past_due=dpd,
        late_fee=late_fee,
        segment=segment,
    )


# Dates of birth count back from this fixed day, not from today. Faker's date_of_birth()
# used today, so every DOB moved by a day each day, and it drew differently on Windows and
# Linux, so the Docker container had different borrowers from the laptop (found 24 Oct).
DOB_ANCHOR = date(2026, 10, 21)


def make_dob(min_age: int = 21, max_age: int = 60) -> date:
    return DOB_ANCHOR - timedelta(days=random.randint(min_age * 365, max_age * 365))


def seed(bind=engine, today: date | None = None) -> None:
    """Drop and recreate all tables, then insert 200 synthetic borrowers + loans.
    The same borrowers on every machine and every day; only due dates follow today, so
    "9 days overdue" stays true whenever the demo runs."""
    random.seed(42)
    Faker.seed(42)  # names and cities: the same data every run, so evals are reproducible
    fake = Faker("en_IN")
    today = today or date.today()

    Base.metadata.drop_all(bind)
    Base.metadata.create_all(bind)

    with Session(bind) as session:
        for segment, count in SEGMENT_COUNTS.items():
            for _ in range(count):
                borrower = Borrower(
                    name=fake.name(),
                    # one uniform format, so "last 4 digits" checks are simple
                    phone=f"+91{random.randint(6, 9)}{random.randint(0, 999_999_999):09d}",
                    dob=make_dob(),
                    preferred_language=random.choices(
                        ["hinglish", "hi", "en"], weights=[60, 25, 15]
                    )[0],
                    city=fake.city(),
                    dnd_flag=random.random() < 0.03,
                )
                session.add(borrower)
                session.flush()  # assigns borrower.id
                session.add(make_loan(segment, borrower.id, today))
        session.commit()
    log.info("Seeded %d borrowers", sum(SEGMENT_COUNTS.values()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    seed()
