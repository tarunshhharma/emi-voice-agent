"""Protection for a public demo link: an access code, a daily call cap, a cap on calls running
at once, and a time limit per call.

Every call spends OpenAI and Sarvam credit, so a public URL must not let strangers start calls.
All are off unless set in the environment (DEMO_ACCESS_CODE, MAX_CALLS_PER_DAY, MAX_ACTIVE_CALLS,
MAX_CALL_SECONDS), so local development is unchanged.
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


def busy(active_calls: int, limit: int | None) -> str | None:
    """Refuse a new call while `limit` calls are already running: each one costs money,
    and the free host has 512 MB of RAM."""
    if limit is not None and active_calls >= limit:
        return "The demo is busy with other calls. Please try again in a few minutes."
    return None


def env_int(name: str) -> int | None:
    """A whole-number setting, or None when it isn't set (the limit is off)."""
    value = os.getenv(name)
    return int(value) if value else None


def calls_today(session: Session) -> int:
    """Spoken calls started since midnight UTC."""
    midnight = datetime.combine(datetime.now(timezone.utc).date(), time.min, tzinfo=timezone.utc)
    return session.scalar(select(func.count()).select_from(Call)
                          .where(Call.channel == "browser", Call.started_at >= midnight)) or 0


def check(session: Session, given_code: str | None) -> str | None:
    cap = env_int("MAX_CALLS_PER_DAY")
    return refusal(given_code, calls_today(session) if cap is not None else 0,
                   access_code=os.getenv("DEMO_ACCESS_CODE") or None, cap=cap)
