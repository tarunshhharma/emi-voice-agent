"""Compare STT providers on the test clips. Usage: python -m evals.wer [--speed 2] [--configs a,b]

For each provider setting and each clip (synthetic clean, synthetic through a simulated phone
line, and your real recordings if present), stream the audio through the same StreamingSTT
client the agent uses and score the final transcript:
  WER    word error rate (word-level edit distance / reference words)
  CER    the same on characters; kinder to spelling variants like "hoon" vs "hun"
  entity share of key words that came through: names, numbers, dates, "mat karna"
  final  ms from end of speech to the final transcript
"""
import argparse
import asyncio
import json
import re
import statistics
import time
import unicodedata
import wave
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from app.pipeline.audio import phone_quality
from app.pipeline.stt.deepgram import DeepgramSTT
from app.pipeline.stt.sarvam import SarvamSTT
from evals.wer_clips import CLIPS, clip_for

load_dotenv()
HERE = Path(__file__).resolve().parent
CLIP_DIRS = {"synthetic": HERE / "clips" / "synthetic", "real": HERE / "clips" / "real"}
RESULTS = HERE / "results"
CHUNK = 3200  # 100 ms

CONFIGS = {
    "sarvam-translit": lambda: SarvamSTT(mode="translit"),
    "sarvam-codemix": lambda: SarvamSTT(mode="codemix"),
    "deepgram-hi": lambda: DeepgramSTT(language="hi"),
    "deepgram-multi": lambda: DeepgramSTT(language="multi"),
}

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


# --- scoring ---


def normalize(text: str) -> str:
    """Lower-case, drop punctuation, unify spelling variants that sound the same."""
    t = unicodedata.normalize("NFC", text).lower().translate(DEVANAGARI_DIGITS)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 3,500 -> 3500
    t = t.replace("़", "")  # nukta: फ़ -> फ
    t = t.replace("ँ", "ं")  # chandrabindu -> anusvara: हाँ -> हां
    t = re.sub(r"[^0-9a-zऀ-ॣ०-ॿ\s]", " ", t)  # punctuation, danda (।)
    return " ".join(t.split())


def edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance: substitutions + deletions + insertions to turn hyp into ref."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def is_devanagari(text: str) -> bool:
    letters = [c for c in text if c.isalpha() or "ऀ" <= c <= "ॿ"]
    return bool(letters) and sum("ऀ" <= c <= "ॿ" for c in letters) / len(letters) > 0.3


def reference_for(clip: dict, hypothesis: str) -> str:
    """Score against the reference written in the same script the provider answered in;
    otherwise every word would count as wrong just for the script."""
    return clip["tts"] if is_devanagari(hypothesis) else clip["roman"]


def entity_hits(clip: dict, hypothesis: str) -> int:
    squashed = normalize(hypothesis).replace(" ", "")
    return sum(any(normalize(v).replace(" ", "") in squashed for v in variants) for variants in clip["entities"])


def score(clip: dict, hypothesis: str) -> dict:
    ref = normalize(reference_for(clip, hypothesis))
    hyp = normalize(hypothesis)
    return {
        "ref_words": len(ref.split()),
        "word_edits": edit_distance(ref.split(), hyp.split()),
        "ref_chars": len(ref.replace(" ", "")),
        "char_edits": edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", ""))),
        "entities": len(clip["entities"]),
        "entity_hits": entity_hits(clip, hypothesis),
    }


# --- running a clip through a provider ---


async def transcribe(stt, pcm: bytes, speed: float, timeout_s: float = 15) -> tuple[str, float]:
    """Send one clip as a single utterance, like the session does. Returns (final text, final ms)."""
    await stt.connect()

    async def first_final() -> str:
        async for transcript in stt.transcripts():
            if transcript.final:
                return transcript.text
        return ""

    reader = asyncio.create_task(first_final())
    try:
        await stt.start_utterance()
        for i in range(0, len(pcm), CHUNK):
            await stt.send_audio(pcm[i:i + CHUNK])
            await asyncio.sleep(0.1 / speed)
        ended = time.monotonic()
        await stt.end_utterance()
        text = await asyncio.wait_for(reader, timeout_s)
        return text, (time.monotonic() - ended) * 1000
    finally:
        reader.cancel()
        await stt.close()


def load_pcm(path: Path) -> bytes:
    with wave.open(str(path)) as f:
        assert f.getframerate() == 16000 and f.getsampwidth() == 2 and f.getnchannels() == 1, path
        return f.readframes(f.getnframes())


def test_cases() -> list[tuple[str, dict, bytes]]:
    cases = []
    for clip in CLIPS:
        synthetic = CLIP_DIRS["synthetic"] / f"{clip['id']}.wav"
        if synthetic.exists():
            pcm = load_pcm(synthetic)
            cases += [("synthetic-clean", clip, pcm), ("synthetic-phone", clip, phone_quality(pcm))]
        real = CLIP_DIRS["real"] / f"{clip['id']}.wav"
        if real.exists():
            cases.append(("real", clip_for("real", clip), load_pcm(real)))
    return cases


async def run_all(configs: list[str], speed: float, concurrency: int, conditions: list[str] | None = None) -> list[dict]:
    limit = asyncio.Semaphore(concurrency)  # providers cap how many streams you may open at once

    async def one(config: str, condition: str, clip: dict, pcm: bytes) -> dict:
        async with limit:
            row = {"config": config, "condition": condition, "clip": clip["id"]}
            try:
                text, final_ms = await transcribe(CONFIGS[config](), pcm, speed)
                row.update(text=text, final_ms=round(final_ms), **score(clip, text))
            except Exception as e:  # one failed clip must not sink the whole run
                row["error"] = f"{type(e).__name__}: {e}"
            return row

    cases = [c for c in test_cases() if not conditions or c[0] in conditions]
    return await asyncio.gather(*(one(c, cond, clip, pcm) for c in configs for cond, clip, pcm in cases))


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["condition"], r["config"]), []).append(r)
    summary = []
    for (condition, config), rs in sorted(groups.items()):
        ok = [r for r in rs if "error" not in r]
        if not ok:
            summary.append({"condition": condition, "config": config, "clips": 0, "errors": len(rs)})
            continue
        summary.append({
            "condition": condition, "config": config, "clips": len(ok), "errors": len(rs) - len(ok),
            # corpus-level rates: total edits over total reference length, not an average of per-clip rates
            "wer": sum(r["word_edits"] for r in ok) / sum(r["ref_words"] for r in ok),
            "cer": sum(r["char_edits"] for r in ok) / sum(r["ref_chars"] for r in ok),
            "entity": sum(r["entity_hits"] for r in ok) / sum(r["entities"] for r in ok),
            "final_ms_p50": statistics.median(r["final_ms"] for r in ok),
        })
    return summary


def markdown_table(summary: list[dict]) -> str:
    lines = ["| Audio | Provider | WER | CER | Key words right | Final (p50) | Clips |",
             "|---|---|---|---|---|---|---|"]
    for s in summary:
        if not s["clips"]:
            lines.append(f"| {s['condition']} | {s['config']} | – | – | – | – | 0 ({s['errors']} errors) |")
            continue
        errors = f" ({s['errors']} errors)" if s["errors"] else ""
        lines.append(f"| {s['condition']} | {s['config']} | {s['wer']:.0%} | {s['cer']:.0%} | "
                     f"{s['entity']:.0%} | {s['final_ms_p50']:.0f} ms | {s['clips']}{errors} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--speed", type=float, default=2.0, help="send audio this many times faster than real time")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--audio", help="only these audio conditions, e.g. real or synthetic-clean,synthetic-phone")
    parser.add_argument("--rescore", nargs="+", metavar="RESULTS_JSON",
                        help="re-score saved transcripts with the current scoring; no API calls")
    args = parser.parse_args()

    if args.rescore:
        clips = {c["id"]: c for c in CLIPS}
        latest: dict[tuple, dict] = {}  # one row per (config, audio, clip): later files win
        for path in args.rescore:
            for r in json.loads(Path(path).read_text(encoding="utf-8"))["rows"]:
                key = (r["config"], r["condition"], r["clip"])
                if "error" in r and key in latest:
                    continue  # a failed re-run doesn't replace an earlier success
                if "error" not in r:
                    r.update(score(clip_for(r["condition"], clips[r["clip"]]), r["text"]))
                latest[key] = r
        print(markdown_table(summarize(list(latest.values()))))
        return

    if not test_cases():
        raise SystemExit("No clips found. Run `python -m evals.make_clips` first, or add real clips to evals/clips/real/.")
    rows = asyncio.run(run_all(args.configs.split(","), args.speed, args.concurrency,
                               args.audio.split(",") if args.audio else None))
    summary = summarize(rows)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"wer-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(markdown_table(summary))
    print(f"\nPer-clip transcripts: {out}")


if __name__ == "__main__":
    main()
