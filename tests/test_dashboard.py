"""The dashboard's numbers, computed on a small known set of calls."""
import json

from app.agent.tools import log_event, start_call
from app.db.models import Outcome, Turn
from dashboard import data


def make_calls(session):
    """Three browser calls on the test loan: a promise to pay, a failed verification with a
    blocked sentence, and one still in progress."""
    ptp = start_call(session, 1, channel="browser")
    ptp.verified, ptp.outcome = True, "PTP"
    session.add(Outcome(call_id=ptp.id, type="PTP", ptp_amount=3500))
    log_event(session, ptp.id, "get_loan_summary", {}, {"ok": True, "emi_amount": 3500})
    session.add_all([
        Turn(call_id=ptp.id, speaker="caller", text="Haan, DOB 14-05-1990", stt_ms=300),
        Turn(call_id=ptp.id, speaker="agent", text="Dhanyavaad.", voice_to_voice_ms=2500, filler_ms=1800),
        Turn(call_id=ptp.id, speaker="agent", text="Theek hai.", voice_to_voice_ms=7000, interrupted=True, barge_in_ms=320),
    ])
    failed = start_call(session, 1, channel="browser")
    failed.outcome = "verify_failed"
    log_event(session, failed.id, "verify_identity", {"factor": "dob"}, {"ok": False, "verified": False})
    log_event(session, failed.id, "guardrail_block", {"reason": "malformed_output"}, {"blocked": "{...}"})
    start_call(session, 1, channel="browser")
    sim = start_call(session, 1, channel="sim")  # must not count when only browser is selected
    sim.outcome = "PTP"
    session.commit()
    return ptp, failed


def test_headline_counts_only_the_selected_channel(session):
    make_calls(session)
    head = data.headline(session, ["browser"])
    assert head["calls"] == 3 and head["verified_pct"] == 33 and head["with_outcome_pct"] == 67
    assert head["ptp_count"] == 1 and head["ptp_amount"] == 3500
    assert head["voice_to_voice_p50"] == 2500


def test_safety_numbers(session):
    make_calls(session)
    s = data.safety(session, ["browser"])
    assert s["disclosures_before_verification"] == 0  # the summary only went to the verified call
    assert s["guardrail_blocks"] == {"malformed_output": 1} and s["failed_verifications"] == 1


def test_loan_data_in_an_unverified_call_is_counted_as_a_disclosure(session):
    call = start_call(session, 1, channel="browser")  # never verified
    log_event(session, call.id, "get_loan_summary", {}, {"ok": True, "emi_amount": 3500})
    assert data.safety(session, ["browser"])["disclosures_before_verification"] == 1


def test_flagged_calls_say_why(session):
    ptp, failed = make_calls(session)
    why = dict(zip(data.flagged(session, ["browser"])["call"], data.flagged(session, ["browser"])["why"]))
    assert why[ptp.id] == "slow turn: 7.0 s"
    assert why[failed.id] == "verification failed, guardrail: malformed_output"
    assert len(why) == 2  # the call in progress has nothing to review


def test_latency_and_barge_ins(session):
    make_calls(session)
    stages = data.latency_by_stage(session).set_index("stage")
    assert stages.loc["voice_to_voice", "n"] == 2 and stages.loc["stt", "p50"] == 300
    assert data.barge_ins(session) == {"count": 1, "p50_ms": 320}
    assert list(data.latency_per_turn(session)["voice_to_voice_ms"]) == [2500, 7000]


def test_transcript_masks_dates_and_phone_numbers(session):
    ptp, _ = make_calls(session)
    said = data.transcript(session, ptp.id)["said"].tolist()
    assert said[0] == "Haan, DOB [date]"
    assert "14-05-1990" not in json.dumps(said)
