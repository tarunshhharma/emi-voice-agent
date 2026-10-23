"""Tune the VAD's end-of-turn silence on recorded turn-taking clips.
Usage: python -m evals.vad_tune [--gain-db 24] [--silence 300,500,700,900]

Each clip in evals/clips/vad/ has a known number of turns: sentences with pauses inside
must stay one turn, two answers 3 s apart must be two, fan or TV noise must be none.
The same fixed gain is applied to every clip, standing in for the browser's auto gain
control (the recordings were made without it). Also reports how much real speech each
clip contains, to check the 300 ms barge-in threshold against short words like "haan".
"""
import argparse
from pathlib import Path

import numpy as np

from app.pipeline.audio import FRAME_BYTES, FRAME_MS
from app.pipeline.session import BARGE_IN_FRAMES
from app.pipeline.vad import SileroVAD
from evals.wer import load_pcm

CLIPS = Path(__file__).resolve().parent / "clips" / "vad"
TAIL = bytes(FRAME_BYTES * 75)  # 1.5 s of silence after each clip, so the last turn can end

# clip id: (turns expected, what was said or played)
EXPECTED = {
    "v01": (1, "DOB with a ~1 s pause in the middle"),
    "v02": (1, "'Main EMI... umm...' then the rest"),
    "v03": (1, "last-4 digits read one by one with pauses"),
    "v04": (1, "'Haan.' alone"),
    "v05": (1, "'Haan haan, theek hai.'"),
    "v06": (1, "long sentence with two pauses"),
    "v07": (1, "fast sentence, no pause"),
    "v08": (1, "quiet 'Ji, main hi bol raha hoon'"),
    "v09": (2, "'Haan.' then 3 s silence then 'Bataiye.'"),
    "n01": (0, "fan or AC noise only"),
    "n02": (0, "TV or music, nobody speaking"),
    "n03": (1, "a sentence over TV noise"),
}


def with_gain(pcm: bytes, gain_db: float) -> bytes:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * 10 ** (gain_db / 20)
    return np.clip(x, -32768, 32767).astype(np.int16).tobytes()


def run_clip(pcm: bytes, silence_ms: int) -> dict:
    """Feed one clip through the VAD like the session does. Returns the turns found and the
    longest run of voiced frames in any turn (what barge-in would count)."""
    vad = SileroVAD(silence_ms=silence_ms)
    audio = pcm + TAIL
    turns, voiced, longest, start = [], 0, 0, None
    for i in range(len(audio) // FRAME_BYTES):
        for event in vad.feed(audio[i * FRAME_BYTES:(i + 1) * FRAME_BYTES]):
            if event == "speech_start":
                start, voiced = i, 0
            elif event == "speech_end":
                turns.append((start * FRAME_MS, i * FRAME_MS))
        if vad.voiced:
            voiced += 1
            longest = max(longest, voiced)
    return {"turns": turns, "voiced_ms": longest * FRAME_MS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gain-db", type=float, default=24)
    parser.add_argument("--silence", default="300,500,700,900")
    args = parser.parse_args()
    settings = [int(s) for s in args.silence.split(",")]

    clips = {cid: with_gain(load_pcm(CLIPS / f"{cid}.wav"), args.gain_db)
             for cid in EXPECTED if (CLIPS / f"{cid}.wav").exists()}
    print("| Clip | Expected | " + " | ".join(f"{s} ms" for s in settings) + " | What it is |")
    print("|---|---|" + "---|" * len(settings) + "---|")
    correct = {s: 0 for s in settings}
    for cid, pcm in clips.items():
        want, what = EXPECTED[cid]
        cells = []
        for s in settings:
            got = len(run_clip(pcm, s)["turns"])
            correct[s] += got == want
            cells.append(f"{got}" + ("" if got == want else " ✗"))
        print(f"| {cid} | {want} | " + " | ".join(cells) + f" | {what} |")
    print("| **correct** | | " + " | ".join(f"**{correct[s]}/{len(clips)}**" for s in settings) + " | |")

    print(f"\nLongest voiced run per clip (barge-in fires at {BARGE_IN_FRAMES * FRAME_MS} ms of voiced speech):")
    for cid, pcm in clips.items():
        ms = run_clip(pcm, 500)["voiced_ms"]
        flag = "would interrupt" if ms >= BARGE_IN_FRAMES * FRAME_MS else "would not interrupt"
        print(f"  {cid}  {ms:5d} ms  {flag}   ({EXPECTED[cid][1]})")


if __name__ == "__main__":
    main()
