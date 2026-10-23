import math
import wave
from array import array

import pytest

from app.pipeline.audio import (
    FRAME_BYTES,
    FRAMES_PER_SECOND,
    SAMPLES_PER_FRAME,
    SILENCE_DB,
    dbfs,
    save_wav,
)


def tone(amplitude: float, samples: int = SAMPLES_PER_FRAME) -> bytes:
    """A 440 Hz sine wave at 16 kHz as 16-bit PCM bytes."""
    return array("h", (int(amplitude * 32767 * math.sin(2 * math.pi * 440 * n / 16000))
                       for n in range(samples))).tobytes()


def test_frame_sizes():
    assert SAMPLES_PER_FRAME == 320
    assert FRAME_BYTES == 640
    assert FRAMES_PER_SECOND == 50


def test_silence_is_the_floor():
    assert dbfs(bytes(FRAME_BYTES)) == SILENCE_DB
    assert dbfs(b"") == SILENCE_DB


def test_full_scale_sine_is_about_minus_3_db():
    # A sine's RMS is peak / sqrt(2), which is -3 dB.
    assert dbfs(tone(1.0)) == pytest.approx(-3.0, abs=0.2)


def test_half_amplitude_is_6_db_quieter():
    assert dbfs(tone(0.5)) - dbfs(tone(1.0)) == pytest.approx(-6.0, abs=0.2)


def test_save_wav_round_trip(tmp_path):
    pcm = tone(0.3, samples=16000)  # one second
    path = save_wav(tmp_path / "sub" / "clip.wav", pcm)
    with wave.open(str(path)) as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 16000)
        assert f.readframes(f.getnframes()) == pcm
