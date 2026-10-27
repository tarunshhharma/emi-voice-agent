"""Protection for a public demo link: an access code and a daily call cap.

Every call spends OpenAI and Sarvam credit, so a public URL must not let strangers start calls.
Both are off unless set in the environment (DEMO_ACCESS_CODE, MAX_CALLS_PER_DAY), so local
development is unchanged.
"""
import hmac
import os
from datetime import datetime, time, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Call


def refusal(given_code: str | None, calls_today: int,
            access_code: str | None = None, cap: int | None = None) -> str | None:
    """Why this visitor may not start a call, or None if they may."""
    if access_code and not hmac.compare_digest(given_code or "", access_code):  # constant-time compare
        return "This demo needs an access code. Ask the person who shared the link."
    if cap is not None and calls_today >= cap:
        return "The demo has reached today's call limit. Please try again tomorrow."
    return None


def calls_today(session: Session) -> int:
    """Spoken calls started since midnight UTC."""
    midnight = datetime.combine(datetime.now(timezone.utc).date(), time.min, tzinfo=timezone.utc)
    return session.scalar(select(func.count()).select_from(Call)
                          .where(Call.channel == "browser", Call.started_at >= midnight)) or 0


def check(session: Session, given_code: str | None) -> str | None:
    cap = os.getenv("MAX_CALLS_PER_DAY")
    return refusal(given_code, calls_today(session) if cap else 0,
                   access_code=os.getenv("DEMO_ACCESS_CODE") or None, cap=int(cap) if cap else None)
