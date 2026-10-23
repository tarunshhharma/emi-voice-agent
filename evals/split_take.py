"""Split one long recording of all 20 WER sentences into c01.wav ... c20.wav.
Usage: python -m evals.split_take evals/clips/real/tarun-all.wav [--silence-ms 700] [--join 7]

Runs the agent's own Silero VAD over the take. A silence longer than --silence-ms ends a
sentence, so it must be longer than pauses inside a sentence but shorter than the gaps
between sentences. Writes nothing unless it finds exactly one segment per sentence.
"""
import argparse
import logging
from pathlib import Path

from app.pipeline.audio import FRAME_BYTES, FRAME_MS, save_wav
from app.pipeline.vad import SileroVAD
from evals.wer import load_pcm
from evals.wer_clips import CLIPS

log = logging.getLogger(__name__)
PAD_FRAMES = 15  # 300 ms kept before each sentence: VAD fires slightly late (same as the pre-roll)


def find_segments(pcm: bytes, vad: SileroVAD) -> list[tuple[int, int]]:
    """Feed the take frame by frame; return (start, end) byte offsets of each speech segment."""
    segments, start = [], None
    n_frames = len(pcm) // FRAME_BYTES
    for i in range(n_frames):
        for event in vad.feed(pcm[i * FRAME_BYTES:(i + 1) * FRAME_BYTES]):
            if event == "speech_start":
                start = max(i - PAD_FRAMES, 0)
            elif event == "speech_end":
                segments.append((start * FRAME_BYTES, (i + 1) * FRAME_BYTES))  # includes the trailing silence
                start = None
    if start is not None:  # take ended mid-speech
        segments.append((start * FRAME_BYTES, n_frames * FRAME_BYTES))
    return segments


def join(segments: list[tuple[int, int]], numbers: list[int]) -> list[tuple[int, int]]:
    """Merge segment n (1-based) with the one after it, for a long pause inside a sentence."""
    segments = list(segments)
    for n in sorted(numbers, reverse=True):  # from the back, so earlier numbers stay valid
        segments[n - 1:n + 1] = [(segments[n - 1][0], segments[n][1])]
    return segments


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("take", type=Path)
    parser.add_argument("--silence-ms", type=int, default=700)
    parser.add_argument("--join", type=int, action="append", default=[],
                        help="merge segment N with the next one (repeatable)")
    args = parser.parse_args()

    pcm = load_pcm(args.take)
    segments = join(find_segments(pcm, SileroVAD(silence_ms=args.silence_ms)), args.join)
    ms_per_byte = FRAME_MS / FRAME_BYTES
    for n, (start, end) in enumerate(segments, 1):
        log.info("%2d  %6.2fs - %6.2fs", n, start * ms_per_byte / 1000, end * ms_per_byte / 1000)
    if len(segments) != len(CLIPS):
        raise SystemExit(f"Found {len(segments)} segments, expected {len(CLIPS)}. Nothing written. "
                         "Too many: raise --silence-ms. Too few: lower it.")
    for clip, (start, end) in zip(CLIPS, segments):
        save_wav(args.take.parent / f"{clip['id']}.wav", pcm[start:end])
    log.info("wrote %d clips to %s", len(segments), args.take.parent)


if __name__ == "__main__":
    main()
