import pytest

from app.pipeline.audio import FRAME_BYTES
from app.pipeline.vad import MODEL_PATH, WINDOW_MS, SileroVAD

FRAME = bytes(FRAME_BYTES)  # content doesn't matter when the probability is faked


def scripted(probs):
    """A fake model: returns the next probability from the list for each 32 ms window."""
    it = iter(probs)
    return lambda window: next(it, 0.0)


def run(vad, frames):
    """Feed frames; return (frame_index, event) for every event."""
    out = []
    for i in range(frames):
        out += [(i, e) for e in vad.feed(FRAME)]
    return out


def test_window_is_32_ms():
    assert WINDOW_MS == 32


def test_speech_then_silence_gives_start_and_end():
    # 10 speech windows, then 20 silent ones (640 ms of silence, more than the 500 ms limit).
    vad = SileroVAD(silence_ms=500, prob_fn=scripted([0.9] * 10 + [0.05] * 20))
    events = [e for _, e in run(vad, 60)]
    assert events == ["speech_start", "speech_end"]


def test_short_pause_does_not_end_the_turn():
    # A 320 ms pause mid-sentence (10 windows) is shorter than 500 ms: still one turn.
    probs = [0.9] * 10 + [0.05] * 10 + [0.9] * 10 + [0.05] * 20
    vad = SileroVAD(silence_ms=500, prob_fn=scripted(probs))
    assert [e for _, e in run(vad, 90)] == ["speech_start", "speech_end"]


def test_shorter_silence_setting_ends_sooner():
    probs = [0.9] * 5 + [0.05] * 30
    end_frame = {}
    for silence_ms in (300, 700):
        vad = SileroVAD(silence_ms=silence_ms, prob_fn=scripted(probs))
        end_frame[silence_ms] = next(i for i, e in run(vad, 60) if e == "speech_end")
    assert end_frame[300] < end_frame[700]


def test_hysteresis_middle_probability_keeps_speech_going():
    # 0.4 is below the start threshold (0.5) but above the stop threshold (0.35):
    # it can't start speech, but it doesn't count as silence either.
    vad = SileroVAD(prob_fn=scripted([0.4] * 5 + [0.9] + [0.4] * 30))
    events = [e for _, e in run(vad, 60)]
    assert events == ["speech_start"]
    assert vad.speaking


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Silero model not downloaded yet")
def test_real_model_hears_no_speech_in_silence():
    vad = SileroVAD()
    assert run(vad, 100) == []  # 2 s of digital silence
