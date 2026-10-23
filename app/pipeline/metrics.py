"""Per-turn latency: when each stage of one caller turn finished, and the gaps between them.
Usage: python -m app.pipeline.metrics   (p50 / p95 per stage over all browser calls)

All times are time.monotonic() on the server. "Caller stopped" is the arrival of their last
voiced frame, so network delay before the server is not included.
"""
import statistics
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Call, SessionLocal, Turn


@dataclass
class TurnTimer:
    caller_stopped: float | None = None  # last voiced frame of the caller's utterance
    speech_end: float | None = None  # VAD decided the turn is over
    stt_final: float | None = None  # final transcript arrived
    agent_start: float | None = None  # agent turn began (can wait for a previous one)
    first_sentence: float | None = None  # first checked sentence ready to speak
    first_audio: float | None = None  # first audio byte sent to the browser
    barge_in_ms: int | None = None  # caller started talking over this reply -> stop sent

    def durations(self) -> dict:
        def gap(start, end):
            return round((end - start) * 1000) if start is not None and end is not None else None

        return {
            "endpoint_ms": gap(self.caller_stopped, self.speech_end),
            "stt_ms": gap(self.speech_end, self.stt_final),
            "llm_ttft_ms": gap(self.agent_start, self.first_sentence),
            "tts_first_byte_ms": gap(self.first_sentence, self.first_audio),
            "voice_to_voice_ms": gap(self.caller_stopped, self.first_audio),
            "barge_in_ms": self.barge_in_ms,
        }


STAGES = ["endpoint_ms", "stt_ms", "llm_ttft_ms", "tts_first_byte_ms", "voice_to_voice_ms", "barge_in_ms"]


def percentile(values: list[int], p: int) -> int:
    """Nearest-rank percentile: p95 of 20 values is the 19th smallest."""
    ordered = sorted(values)
    return ordered[max(0, -(-p * len(ordered) // 100) - 1)]


def summary(session: Session, channel: str = "browser") -> dict:
    turns = session.scalars(select(Turn).join(Call, Turn.call_id == Call.id).where(Call.channel == channel)).all()
    out = {}
    for stage in STAGES:
        speaker = "caller" if stage == "stt_ms" else "agent"  # STT time belongs to what the caller said
        values = [v for t in turns if t.speaker == speaker and (v := getattr(t, stage)) is not None]
        if values:
            out[stage] = {"n": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95),
                          "mean": round(statistics.mean(values))}
    return out


def main() -> None:
    with SessionLocal() as session:
        rows = summary(session)
    if not rows:
        print("No browser turns with timings yet. Have a call at http://localhost:8000 first.")
        return
    print("| Stage | n | p50 | p95 | mean |\n|---|---|---|---|---|")
    for stage, s in rows.items():
        print(f"| {stage} | {s['n']} | {s['p50']} ms | {s['p95']} ms | {s['mean']} ms |")


if __name__ == "__main__":
    main()
