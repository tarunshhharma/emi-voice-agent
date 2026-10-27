from app.agent.tools import start_call
from app.demo_gate import calls_today, refusal


def test_local_development_has_no_gate():
    assert refusal(None, calls_today=999) is None  # no code, no cap configured


def test_access_code_must_match():
    assert refusal(None, 0, access_code="secret") is not None
    assert refusal("wrong", 0, access_code="secret") is not None
    assert refusal("secret", 0, access_code="secret") is None


def test_daily_cap():
    assert refusal("secret", 19, access_code="secret", cap=20) is None
    assert "limit" in refusal("secret", 20, access_code="secret", cap=20)


def test_only_todays_spoken_calls_count(session):
    from datetime import datetime, timedelta, timezone
    from app.db.models import Call
    start_call(session, 1, channel="browser")
    start_call(session, 1, channel="sim")  # evals don't use up the demo's calls
    old = start_call(session, 1, channel="browser")
    old.started_at = datetime.now(timezone.utc) - timedelta(days=2)
    session.commit()
    assert calls_today(session) == 1
