"""The full eval: every persona several times, each call scored by code and by the LLM judge.
Usage: python -m evals.eval_run [--runs 4] [--workers 4] [persona_id ...]

Each run of a persona calls a different borrower of its segment. Calls run in parallel threads,
each with its own DB session. Results go to evals/results/eval-<time>.json and a table.
"""
import argparse
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from app.db.models import SessionLocal
from evals.judge import judge
from evals.personas import PERSONAS
from evals.simulate import simulate

RESULTS = Path(__file__).resolve().parent / "results"
log = logging.getLogger(__name__)


def jobs(personas: list[dict], runs: int) -> list[tuple[dict, int]]:
    """(persona, nth borrower of its pool). Personas sharing a pool (segment + language) never
    share a borrower; an English-speaking persona draws only from English-preference borrowers."""
    pools: dict[tuple, list[str]] = {}
    for p in personas:
        pools.setdefault((p["segment"], p.get("language")), []).append(p["id"])
    out = []
    for run in range(runs):
        for p in personas:
            peers = pools[(p["segment"], p.get("language"))]
            out.append((p, run * len(peers) + peers.index(p["id"])))
    return out


def one_call(client, persona: dict, nth: int) -> dict:
    with SessionLocal() as session:
        try:
            row = simulate(client, session, persona, nth, verbose=False)
            row["judge"] = judge(client, session, row["call_id"])
        except Exception as e:  # one broken call must not sink the run
            log.warning("%s #%d failed: %r", persona["id"], nth, e)
            return {"persona": persona["id"], "error": f"{type(e).__name__}: {e}"}
    j = row["judge"]
    row["compliance_ok"] = j["compliance"] == "pass"
    # A disclosure counts if either check finds one: the code check (loan tool returned data to a
    # caller who must never hear it) or the judge (something was said before verification).
    row["no_disclosure"] = row["disclosure_ok"] and not j["disclosed_before_verification"]
    row["passed"] = row["outcome_ok"] and row["no_disclosure"] and row["compliance_ok"]
    return row


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    def share(key):
        return round(100 * sum(r[key] for r in ok) / len(ok)) if ok else 0
    return {
        "calls": len(rows), "errors": len(rows) - len(ok),
        "task_success_pct": share("outcome_ok"),
        "passed_pct": share("passed"),
        "disclosure_violations": sum(not r["no_disclosure"] for r in ok),
        "compliance_violations": sum(not r["compliance_ok"] for r in ok),
        "judge_outcome_disagreements": sum((r["judge"]["outcome_fits_caller"] == "pass") != r["outcome_ok"] for r in ok),
        "filter_blocks": sum(r["guardrail_blocks"] for r in ok),
        "avg_turns": round(statistics.mean(r["turns"] for r in ok), 1) if ok else 0,
        "avg_quality": round(statistics.mean(r["judge"]["quality"] for r in ok), 1) if ok else 0,
    }


def table(rows: list[dict]) -> str:
    lines = ["| Persona | Passed | Outcomes | Avg turns | Quality | Problems |", "|---|---|---|---|---|---|"]
    for pid in dict.fromkeys(r["persona"] for r in rows):
        mine = [r for r in rows if r["persona"] == pid]
        ok = [r for r in mine if "error" not in r]
        outcomes = ", ".join(str(r["outcome"]) for r in ok) or "–"
        problems = [r["judge"]["reason"] for r in ok if not r["passed"]] + [r["error"] for r in mine if "error" in r]
        turns = f"{statistics.mean(r['turns'] for r in ok):.1f}" if ok else "–"
        quality = f"{statistics.mean(r['judge']['quality'] for r in ok):.1f}" if ok else "–"
        lines.append(f"| {pid} | {sum(r['passed'] for r in ok)}/{len(mine)} | {outcomes} | {turns} | {quality} | "
                     + ("; ".join(problems)[:300] or "–") + " |")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("personas", nargs="*")
    args = parser.parse_args()
    personas = [p for p in PERSONAS if not args.personas or p["id"] in args.personas]

    client = OpenAI()
    todo = jobs(personas, args.runs)
    print(f"{len(todo)} calls ({len(personas)} personas x {args.runs} runs), {args.workers} at a time...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda job: one_call(client, *job), todo))

    summary = summarize(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str), encoding="utf-8")
    print(table(rows))
    print("\n" + json.dumps(summary, indent=1))
    print(f"\nDetails: {out}")


if __name__ == "__main__":
    main()
