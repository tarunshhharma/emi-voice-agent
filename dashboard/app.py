"""Calls dashboard. Usage: streamlit run dashboard/app.py

Reads emi.db (read-only). All numbers come from dashboard/data.py; this file is only layout.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "app" and "dashboard" import when run by streamlit

from app.db.models import SessionLocal  # noqa: E402
from dashboard import data  # noqa: E402

st.set_page_config(page_title="EMI Voice Agent · Calls", layout="wide")
st.title("EMI Voice Agent · Calls")

with st.sidebar:
    channels = st.multiselect("Calls from", ["browser", "sim", "text"], default=["browser"],
                              help="browser = spoken calls · sim = simulated callers (evals) · text = terminal chat")
    st.button("Refresh")  # any click re-runs the page against the current database
    st.caption("Synthetic data only. Phone numbers and dates are masked in transcripts.")
if not channels:
    st.info("Pick at least one source of calls in the sidebar.")
    st.stop()

with SessionLocal() as session:
    head = data.headline(session, channels)
    outcomes = data.outcomes(session, channels)
    stages = data.latency_by_stage(session)
    per_turn = data.latency_per_turn(session)
    safety = data.safety(session, channels)
    barge = data.barge_ins(session)
    flagged = data.flagged(session, channels)
    all_calls = data.calls(session, channels)

# --- headline ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Calls", head["calls"])
c2.metric("Verified", f"{head['verified_pct']}%")
c3.metric("Ended with an outcome", f"{head['with_outcome_pct']}%")
c4.metric("Promised to pay", f"₹{head['ptp_amount']:,}", f"{head['ptp_count']} promises", delta_color="off")
v2v = head["voice_to_voice_p50"]
c5.metric("Voice-to-voice p50", f"{v2v / 1000:.1f} s" if v2v else "–", "target < 1.5 s", delta_color="off")

left, right = st.columns(2)
with left:
    st.subheader("Outcomes")
    st.bar_chart(outcomes, x="outcome", y="calls", horizontal=True)
with right:
    st.subheader("Compliance")
    s1, s2, s3 = st.columns(3)
    s1.metric("Loan data to unverified callers", safety["disclosures_before_verification"],
              help="Independent of the output filter: did get_loan_summary ever return data in a call that was never verified? Must be 0.")
    s2.metric("Sentences blocked", sum(safety["guardrail_blocks"].values()))
    s3.metric("Failed verifications", safety["failed_verifications"])
    if safety["guardrail_blocks"]:
        st.caption("Blocked by reason: " + ", ".join(f"{k} {v}" for k, v in safety["guardrail_blocks"].items()))
    st.caption(f"Tools refused by the state machine: {safety['refused_tools']} · deferred until the caller answers: {safety['deferred_tools']}")

# --- latency ---
st.subheader("Latency (spoken calls)")
if stages.empty:
    st.info("No timed turns yet. Make a call at http://localhost:8000.")
else:
    l1, l2 = st.columns([2, 3])
    with l1:
        st.dataframe(stages, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="%d ms") for c in ["p50", "p95", "mean"]})
        st.caption(f"Barge-ins: {barge['count']}" + (f", stop sent after {barge['p50_ms']} ms (p50; target < 300 ms)" if barge["p50_ms"] else ""))
    with l2:
        st.line_chart(per_turn, x="turn", y=["voice_to_voice_ms", "filler_ms"], y_label="ms", x_label="turn (oldest → newest)")
        st.caption("voice_to_voice: caller stops → first real answer. filler: caller stops → first sound, on turns that call a tool.")

# --- flagged ---
st.subheader("Calls to review")
if flagged.empty:
    st.success("Nothing flagged.")
else:
    st.dataframe(flagged, hide_index=True, width="stretch")

# --- one call ---
st.subheader("Call detail")
if all_calls.empty:
    st.info("No calls yet.")
else:
    labels = {int(r.call): f"#{r.call} · {r.borrower} · {r.outcome or 'no outcome'} · {r.started:%d %b %H:%M}" for r in all_calls.itertuples()}
    chosen = st.selectbox("Call", list(labels), format_func=labels.get)
    with SessionLocal() as session:
        t = data.transcript(session, chosen)
        ev = data.tool_events(session, chosen)
    d1, d2 = st.columns([3, 2])
    with d1:
        st.caption("Transcript")
        st.dataframe(t, hide_index=True, width="stretch")
    with d2:
        st.caption("Tool calls, in order (the audit log)")
        st.dataframe(ev, hide_index=True, width="stretch")
