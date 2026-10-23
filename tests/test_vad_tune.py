import numpy as np

from evals.vad_tune import EXPECTED, with_gain


def test_gain_scales_and_clips_instead_of_wrapping():
    pcm = np.array([1000, -1000, 30000], dtype=np.int16).tobytes()
    out = np.frombuffer(with_gain(pcm, 20), dtype=np.int16)  # x10
    assert list(out) == [10000, -10000, 32767]


def test_every_turn_taking_clip_has_an_expected_answer():
    assert {c for c in EXPECTED if c.startswith("n")} == {"n01", "n02", "n03"}
    assert EXPECTED["v09"][0] == 2 and EXPECTED["n01"][0] == 0
